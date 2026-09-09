# This file is part of ts_weathernbeats.
#
# Developed for the Vera C. Rubin Observatory Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Standalone retraining: regenerate the NBEATSx + per-slot Ridge artifacts.

This is the *only* place fitting happens.  It trains the direct-temperature
NBEATSx network (target = absolute ``y``) and a **new per-solar-slot Ridge
layer**: one ``StandardScaler`` + ``Ridge`` corrector per solar-grid slot,
keyed by the target solar coordinate ``phi``.  The two stages are written as a
single self-describing :class:`~lsst.ts.weathernbeats.artifacts.ModelBundle`.

Training telemetry comes from a CSV (to reproduce the SPIE fixture) or, when no
path is given, from the live EFD via
`~lsst.ts.weathernbeats.feature_builder.FeatureBuilder`.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import ModelBundle, SlotModel
from .feature_builder import (
    NB_FUTR,
    NB_HIST,
    NBEATS_HORIZON,
    NBEATS_INPUT_SIZE,
    RIDGE_FEATS,
    FeatureBuilder,
)
from .solar_grid import STEPS_PER_DAY
from .solar_slots import phi_to_equinox_label, snap_to_slot, solar_slots

__all__ = ["load_training_grid", "train_nbeatsx", "fit_slot_ridges", "retrain", "main"]

NBEATS_MAX_STEPS = 700
NBEATS_WIDTH = 16
RIDGE_ALPHA = 1.0
SOLAR_GRID_FREQ = "1800s"
MIN_SLOT_SAMPLES = 30


def load_training_grid(csv_path: str | Path | None) -> pd.DataFrame:
    """Build the solar-grid feature frame from a CSV or the live EFD."""
    if csv_path is not None:
        raw = pd.read_csv(csv_path, comment="#", low_memory=False)
        raw["ds"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_localize(None)
        raw["y"] = raw["mean"]
        if {"max", "min"}.issubset(raw.columns):
            spread = raw["max"] - raw["min"]
            raw.loc[spread > 3.0, "y"] = np.nan
        raw["y"] = raw["y"].ffill().bfill()
        frame = raw[["ds", "y"]].dropna().drop_duplicates("ds").sort_values("ds")
        return FeatureBuilder.build_grid(frame)
    fb = FeatureBuilder(simulation_mode=1)
    return asyncio.run(fb.get_features())


def train_nbeatsx(grid: pd.DataFrame):
    """Train the direct-temperature NBEATSx network (target = absolute ``y``)."""
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import HuberLoss
    from neuralforecast.models import NBEATSx

    hist = [c for c in NB_HIST if c in grid.columns]
    futr = [c for c in NB_FUTR if c in grid.columns]
    train = grid[["ds", "y"] + hist + futr].dropna().copy()
    train["unique_id"] = "temp"

    model = NBEATSx(
        h=NBEATS_HORIZON,
        input_size=NBEATS_INPUT_SIZE,
        max_steps=NBEATS_MAX_STEPS,
        hist_exog_list=hist,
        futr_exog_list=futr,
        activation="SELU",
        loss=HuberLoss(),
        learning_rate=0.001,
        batch_size=48,
        scaler_type="identity",
        enable_progress_bar=False,
        enable_model_summary=False,
        stack_types=["trend", "seasonality", "identity", "exogenous"],
        mlp_units=4 * [[NBEATS_WIDTH, NBEATS_WIDTH]],
        n_blocks=[1, 1, 1, 1],
        early_stop_patience_steps=10,
        val_check_steps=50,
    )
    nf = NeuralForecast(models=[model], freq=SOLAR_GRID_FREQ)
    nf.fit(train, val_size=max(NBEATS_HORIZON, int(len(train) * 0.1)))
    return nf, hist, futr


def _nbeatsx_trajectories(nf, grid, hist, futr, stride=1, batch=64):
    """Yield ``(iss, trajectory)`` for sliding issuance points over the grid.

    ``trajectory`` is the NBEATSx absolute-temperature forecast for the
    ``NBEATS_HORIZON`` steps after issuance index ``iss``.
    """
    allx = hist + futr
    n = len(grid)
    requests = [
        iss
        for iss in range(NBEATS_INPUT_SIZE, n - NBEATS_HORIZON, stride)
        if not grid.iloc[iss - NBEATS_INPUT_SIZE : iss][allx].isna().any().any()
        and not grid.iloc[iss : iss + NBEATS_HORIZON][futr].isna().any().any()
    ]
    for b0 in range(0, len(requests), batch):
        hists, futrs, meta = [], [], []
        for iss in requests[b0 : b0 + batch]:
            uid = f"p_{iss}"
            hd = grid.iloc[iss - NBEATS_INPUT_SIZE : iss][["ds", "y"] + allx].copy()
            fd = grid.iloc[iss : iss + NBEATS_HORIZON][["ds"] + futr].copy()
            hd["unique_id"] = uid
            fd["unique_id"] = uid
            hists.append(hd)
            futrs.append(fd)
            meta.append((uid, iss))
        fc = nf.predict(
            df=pd.concat(hists, ignore_index=True),
            futr_df=pd.concat(futrs, ignore_index=True),
        ).reset_index()
        col = next(c for c in fc.columns if c not in ("unique_id", "ds", "index"))
        for uid, iss in meta:
            traj = fc[fc["unique_id"] == uid].sort_values("ds")[col].to_numpy()
            if len(traj) == NBEATS_HORIZON:
                yield iss, traj


def fit_slot_ridges(nf, grid, hist, futr, n_slots=STEPS_PER_DAY, stride=1):
    """Fit one ``StandardScaler`` + ``Ridge`` per solar slot (keyed by phi).

    For every issuance point and every horizon step, the forecast step lands on
    a target solar coordinate; samples are bucketed by the *target* slot phi.
    Each sample is ``(issuance-time features + T_nb at the step) -> measured T``.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    slots = solar_slots(n_slots)
    feat_cols = [c for c in RIDGE_FEATS if c in grid.columns]
    buckets: dict[float, dict[str, list]] = {
        float(p): {"X": [], "y": []} for p in slots
    }

    solar = grid["SolarTime"].to_numpy()
    yv = grid["y"].to_numpy()
    feat_mat = grid[feat_cols].to_numpy(dtype=float)

    for iss, traj in _nbeatsx_trajectories(nf, grid, hist, futr, stride=stride):
        iss_feats = feat_mat[iss]
        if np.any(np.isnan(iss_feats)):
            continue
        for k in range(NBEATS_HORIZON):
            tgt = iss + k
            if np.isnan(yv[tgt]):
                continue
            _, slot_phi = snap_to_slot(float(solar[tgt]), slots)
            buckets[slot_phi]["X"].append(np.concatenate([iss_feats, [traj[k]]]))
            buckets[slot_phi]["y"].append(yv[tgt])

    slot_models: dict[float, SlotModel] = {}
    for phi, data in buckets.items():
        if len(data["X"]) < MIN_SLOT_SAMPLES:
            continue
        x_arr = np.asarray(data["X"])
        y_arr = np.asarray(data["y"])
        scaler = StandardScaler().fit(x_arr)
        ridge = Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(x_arr), y_arr)
        residuals = y_arr - ridge.predict(scaler.transform(x_arr))
        slot_models[phi] = SlotModel(
            phi=phi,
            equinox_label=phi_to_equinox_label(phi),
            feature_list=feat_cols,
            scaler=scaler,
            ridge=ridge,
            residual_std=float(np.std(residuals)),
        )
    if not slot_models:
        raise RuntimeError("No slot had enough samples to fit a Ridge corrector.")
    return slot_models


def retrain(
    csv_path: str | Path | None,
    output: str | Path,
    n_slots: int = STEPS_PER_DAY,
    stride: int = 1,
) -> ModelBundle:
    """Run the full retrain and write the bundle to ``output``."""
    import sklearn

    grid = load_training_grid(csv_path)
    nf, hist, futr = train_nbeatsx(grid)
    slots = fit_slot_ridges(nf, grid, hist, futr, n_slots=n_slots, stride=stride)

    metadata = {
        "model_version": "0.1.0",
        "formulation": "direct-temperature, per-solar-slot Ridge",
        "training_window": [
            str(grid["ds_real"].min()),
            str(grid["ds_real"].max()),
        ],
        "n_slots": n_slots,
        "nbeats_input_size": NBEATS_INPUT_SIZE,
        "nbeats_horizon": NBEATS_HORIZON,
        "nb_hist": hist,
        "nb_futr": futr,
        "ridge_features": next(iter(slots.values())).feature_list,
        "sklearn_version": sklearn.__version__,
    }
    bundle = ModelBundle(nbeatsx=nf, slots=slots, metadata=metadata)
    bundle.save(output)
    return bundle


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for retraining."""
    parser = argparse.ArgumentParser(description="Retrain ts_weathernbeats artifacts.")
    parser.add_argument(
        "--csv", default=None, help="Training telemetry CSV (default: live EFD)."
    )
    parser.add_argument(
        "--output", required=True, help="Output bundle directory."
    )
    parser.add_argument(
        "--n-slots", type=int, default=STEPS_PER_DAY, help="Solar slots per day."
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="Issuance-point stride (grid steps)."
    )
    args = parser.parse_args(argv)
    retrain(args.csv, args.output, n_slots=args.n_slots, stride=args.stride)


if __name__ == "__main__":
    main()
