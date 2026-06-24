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

"""Inference-only two-stage NBEATSx + per-slot Ridge temperature forecaster.

This module loads pre-trained artifacts and predicts; it never fits or refits
anything at inference time.  The inference path (see SPIE paper §4) is:

1. Stage 1 -- run the pre-trained NBEATSx network forward from the 48-step
   solar-grid lookback to obtain a continuous **absolute** temperature forecast
   ``T_NB(phi)`` at every step over the horizon.
2. Stage 2 -- for each forecast step, snap its solar slot ``phi`` to the nearest
   trained slot, build that slot's feature vector (solar-grid features +
   ``T_NB`` at the step) and apply the per-slot ``StandardScaler`` + ``Ridge``
   corrector, which outputs absolute temperature directly.

The result is a bias-corrected continuous curve; the operational twilight, 3 h
dome-opening and 9 h morning values are read off this curve.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import ModelBundle
from .feature_builder import NB_FUTR, NB_HIST, NBEATS_HORIZON, NBEATS_INPUT_SIZE
from .solar_grid import STEPS_PER_DAY, SUN_ALT_MIDPOINT, get_sun_altitude
from .solar_slots import snap_to_slot

__all__ = [
    "WeatherForecastModel",
    "interp_at",
    "DELTA_TIME",
    "DOME_OPENING_LEAD_H",
    "MORNING_HVAC_LEAD_H",
    "STD_LEAD_POLY",
]

# Seconds between successive samples of the published hourlyTrend trajectory
# (matches twilight_forecast_history.ipynb).
DELTA_TIME = 5 * 60

# Operational lead times read off the corrected curve (SPIE paper §3-4):
# 3 h is the last safe M1M3 / dome-opening setpoint change; 9 h is the morning
# HVAC dome-air setpoint.
DOME_OPENING_LEAD_H = 3.0
MORNING_HVAC_LEAD_H = 9.0

# Forecast uncertainty (1-sigma, deg C) as a 5th-order polynomial in the
# forecast lead measured in *solar-grid steps* (2 steps = 1 h).  Fitted to the
# per-lead std of the NBEATSx-Ridge errors from the SPIE evaluation
# (RubinsOraclePaper/results/paper_results_final.csv); see fig3.  Highest power
# first.  The bundle metadata may override this under key "std_lead_poly".
STD_LEAD_POLY = [
    8.02e-07, -6.7537e-05, 0.002195727, -0.033996664, 0.275156312, -0.044874836,
]


def interp_at(
    temperatures: np.ndarray,
    sndstamp_unix: float,
    target_unix: float,
    delta_time: float = DELTA_TIME,
) -> float:
    """Linearly interpolate a uniform-cadence forecast trajectory at a time.

    Matches the ``interp_at`` helper in ``twilight_forecast_history.ipynb``:
    ``temperatures`` is sampled every ``delta_time`` seconds starting one step
    after ``sndstamp_unix``.  Returns ``NaN`` when ``target_unix`` falls outside
    the trajectory.
    """
    temperatures = np.asarray(temperatures, dtype=float)
    index_float = (target_unix - sndstamp_unix) / delta_time - 1
    if index_float < 0 or index_float > len(temperatures) - 1:
        return float("nan")
    return float(np.interp(index_float, np.arange(len(temperatures)), temperatures))


class WeatherForecastModel:
    """Load pre-trained artifacts and produce calibrated temperature forecasts.

    Parameters
    ----------
    bundle : `ModelBundle`
        The loaded two-stage model (NBEATSx + per-slot Ridge correctors).
    """

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle
        self._slot_phis = np.array(bundle.slot_phis, dtype=float)

    @classmethod
    def load(cls, path: str | Path) -> "WeatherForecastModel":
        """Load a model from a bundle directory."""
        return cls(ModelBundle.load(path))

    # ── Stage 1: NBEATSx forward pass ─────────────────────────────────────

    def _nbeatsx_forecast(self, grid: pd.DataFrame) -> pd.DataFrame:
        """Run NBEATSx forward from the most recent lookback window.

        Returns the horizon rows with the continuous NBEATSx prediction in a
        ``T_nb`` column alongside the future solar coordinates.
        """
        hist_cols = [c for c in NB_HIST if c in grid.columns]
        futr_cols = [c for c in NB_FUTR if c in grid.columns]

        usable = grid.dropna(subset=hist_cols + ["y"]).reset_index(drop=True)
        if len(usable) < NBEATS_INPUT_SIZE:
            raise ValueError(
                f"Need >= {NBEATS_INPUT_SIZE} clean lookback rows, "
                f"got {len(usable)}."
            )
        hist = usable.iloc[-NBEATS_INPUT_SIZE:][["ds", "y"] + hist_cols + futr_cols].copy()
        hist["unique_id"] = "temp"

        # Future stub: continue the synthetic solar clock for the horizon.
        step = hist["ds"].iloc[-1] - hist["ds"].iloc[-2]
        future_ds = hist["ds"].iloc[-1] + step * np.arange(1, NBEATS_HORIZON + 1)
        futr = self._future_solar_frame(usable, future_ds, futr_cols)

        forecast = self.bundle.nbeatsx.predict(df=hist, futr_df=futr).reset_index(drop=True)
        model_col = next(
            c for c in forecast.columns if c not in ("unique_id", "ds", "index")
        )
        futr = futr.reset_index(drop=True)
        futr["T_nb"] = forecast[model_col].to_numpy()
        return futr

    @staticmethod
    def _future_solar_frame(
        grid: pd.DataFrame, future_ds: np.ndarray, futr_cols: list[str]
    ) -> pd.DataFrame:
        """Build the NBEATSx future-exog frame, extrapolating solar coords.

        The synthetic solar clock is perfectly regular, so the future
        ``SolarTime`` is the lookback's last value advanced by the grid step,
        and the deterministic phase features follow from it.
        """
        n = len(future_ds)
        last_phi = float(grid["SolarTime"].iloc[-1])
        n_steps = len(grid)
        # Step in solar fraction per grid row (1/48 by construction).
        dphi = float(grid["SolarTime"].iloc[-1] - grid["SolarTime"].iloc[-2]) % 1.0
        phi = (last_phi + dphi * np.arange(1, n + 1)) % 1.0

        out = pd.DataFrame({"unique_id": "temp", "ds": future_ds})
        out["SolarTime"] = phi
        if "solar_sin" in futr_cols:
            out["solar_sin"] = np.sin(2 * np.pi * phi)
        if "solar_cos" in futr_cols:
            out["solar_cos"] = np.cos(2 * np.pi * phi)
        # Day-of-year barely changes over a 12 h horizon; carry the last value.
        if "doy_sin" in futr_cols:
            out["doy_sin"] = float(grid["doy_sin"].iloc[-1])
        if "doy_cos" in futr_cols:
            out["doy_cos"] = float(grid["doy_cos"].iloc[-1])
        _ = n_steps  # retained for clarity; grid length not otherwise needed
        return out

    # ── Stage 2: per-slot Ridge correction ───────────────────────────────

    def _correct(self, grid: pd.DataFrame, horizon: pd.DataFrame) -> pd.DataFrame:
        """Apply the per-slot Ridge correctors over the whole horizon curve.

        For each horizon step the slot is selected by the **target** solar
        coordinate ``phi``; the feature vector is the issuance-time solar-grid
        features (last clean lookback row) with the step's ``T_nb`` appended.
        """
        issuance = grid.dropna(subset=["y"]).iloc[-1]
        corrected = np.empty(len(horizon))
        slot_phi_used = np.empty(len(horizon))

        for i, (_, row) in enumerate(horizon.iterrows()):
            _, slot_phi = snap_to_slot(float(row["SolarTime"]), self._slot_phis)
            slot = self.bundle.slots[slot_phi]
            feats = issuance[slot.feature_list].to_numpy(dtype=float)
            x = np.concatenate([feats, [row["T_nb"]]]).reshape(1, -1)
            corrected[i] = float(slot.ridge.predict(slot.scaler.transform(x))[0])
            slot_phi_used[i] = slot_phi

        # Uncertainty band grows with the forecast lead (solar-grid steps from
        # issuance): horizon row i is lead step i+1.  Evaluate the fitted poly
        # (bundle metadata may override the default coefficients).
        lead_steps = np.arange(1, len(horizon) + 1, dtype=float)
        coeffs = self.bundle.metadata.get("std_lead_poly", STD_LEAD_POLY)
        band = np.clip(np.polyval(coeffs, lead_steps), 0.0, None)

        out = horizon[["ds", "SolarTime", "T_nb"]].copy()
        out["slot_phi"] = slot_phi_used
        out["T_forecast"] = corrected
        out["T_std"] = band
        return out.reset_index(drop=True)

    # ── Public API ────────────────────────────────────────────────────────

    def predict(self, grid: pd.DataFrame) -> pd.DataFrame:
        """Return the calibrated continuous forecast curve.

        Parameters
        ----------
        grid : `pandas.DataFrame`
            Solar-grid feature frame from
            `~lsst.ts.weathernbeats.feature_builder.FeatureBuilder`.

        Returns
        -------
        `pandas.DataFrame`
            Columns ``ds`` (synthetic solar clock), ``ds_real`` (real UTC),
            ``SolarTime``, ``T_nb`` (Stage-1 NBEATSx) and ``T_forecast``
            (Stage-2 corrected absolute temperature).
        """
        horizon = self._nbeatsx_forecast(grid)
        curve = self._correct(grid, horizon)
        # Map the solar-grid horizon back onto real UTC.  A solar step is uniform
        # in *solar* time but NOT in wall-clock: daytime steps are short and
        # nighttime steps long (and the factor flips at sunrise/sunset).  Use the
        # actual sun events as anchors -- consecutive events (sunrise->sunset->
        # sunrise) are each +0.5 solar-day-fraction regardless of their real
        # duration, so interpolating solar position against event times yields
        # the correct variable wall-clock step automatically.
        clean = grid.dropna(subset=["y"])
        last_real = pd.Timestamp(clean["ds_real"].iloc[-1])
        ds_real = self._solar_to_real(last_real, len(curve))
        curve.insert(1, "ds_real", ds_real)
        return curve

    @staticmethod
    def _solar_to_real(last_real: pd.Timestamp, n_steps: int) -> np.ndarray:
        """Real-UTC timestamps for ``n_steps`` solar-grid steps past ``last_real``.

        Builds sun-event anchors (sunrise / sunset zero-crossings of the solar
        altitude) over a window bracketing the horizon, assigns each event a
        solar coordinate (+0.5 per event), then interpolates: ``last_real`` ->
        its solar coordinate, advance by ``1/STEPS_PER_DAY`` per step, and map
        each back to real time.  Night steps therefore stretch and day steps
        compress, exactly as the true geometry requires.
        """
        # Minute-cadence sun altitude over [-18 h, +30 h] around the issuance.
        win = pd.date_range(
            last_real - pd.Timedelta(hours=18),
            last_real + pd.Timedelta(hours=30),
            freq="min",
        )
        alt = get_sun_altitude(win)
        prev = np.concatenate([[alt[0]], alt[:-1]])
        ev_idx = np.where(((prev < 0) & (alt >= 0)) | ((prev >= 0) & (alt < 0)))[0]
        ev_real = win[ev_idx].astype("int64").to_numpy() / 1e9  # seconds
        # Consecutive events are +0.5 in solar-day fraction (sunrise<->sunset).
        ev_solar = 0.5 * np.arange(len(ev_real))

        last_s = float(last_real.timestamp())
        solar_now = np.interp(last_s, ev_real, ev_solar)
        step = 1.0 / STEPS_PER_DAY
        solar_fc = solar_now + step * np.arange(1, n_steps + 1)
        real_fc = np.interp(solar_fc, ev_solar, ev_real)
        return pd.to_datetime((real_fc * 1e9).astype("int64"))

    def predict_temperature_at_time(
        self, curve: pd.DataFrame, target: pd.Timestamp | float
    ) -> float:
        """Interpolate the corrected curve at an arbitrary target time.

        ``target`` may be a `pandas.Timestamp` (UTC) or a Unix timestamp.
        Returns ``NaN`` if the target falls outside the forecast horizon.
        """
        target_unix = (
            float(target)
            if isinstance(target, (int, float))
            else pd.Timestamp(target).timestamp()
        )
        x = pd.to_datetime(curve["ds_real"]).astype("int64").to_numpy() / 1e9
        y = curve["T_forecast"].to_numpy(dtype=float)
        if target_unix < x[0] or target_unix > x[-1]:
            return float("nan")
        return float(np.interp(target_unix, x, y))

    def _twilight_time(self, curve: pd.DataFrame) -> pd.Timestamp:
        """Find the evening twilight time within the forecast horizon.

        Twilight is the descending crossing of the sun-altitude midpoint
        (``SUN_ALT_MIDPOINT``) over the curve's real-UTC times; linear
        interpolation refines it between grid steps.  Falls back to the curve's
        sunset (``SolarTime`` nearest 0.5) when no crossing is bracketed.
        """
        ds_real = pd.to_datetime(curve["ds_real"].to_numpy())
        alt = get_sun_altitude(pd.DatetimeIndex(ds_real))
        adj = alt - SUN_ALT_MIDPOINT
        setting = np.where((adj[:-1] >= 0) & (adj[1:] < 0))[0]
        if len(setting) == 0:
            # No crossing in-horizon: use the step closest to sunset (phi=0.5).
            i = int(np.argmin(np.abs(curve["SolarTime"].to_numpy() - 0.5)))
            return pd.Timestamp(ds_real[i])
        i = int(setting[0])
        t0, t1 = ds_real[i].value, ds_real[i + 1].value
        frac = adj[i] / (adj[i] - adj[i + 1])
        return pd.Timestamp(int(t0 + frac * (t1 - t0)))

    def operational_forecast(self, grid: pd.DataFrame) -> dict[str, object]:
        """Produce the corrected curve plus the operational read-off values.

        Returns the continuous Stage-2 curve together with the twilight time
        and the three first-class operational temperatures derived from it:
        the twilight value, the 3 h dome-opening / M1M3 setpoint lead and the
        9 h morning-HVAC lead (each read ``lead`` hours *before* twilight).

        Returns
        -------
        `dict`
            Keys ``curve`` (`pandas.DataFrame`), ``twilight_time``
            (`pandas.Timestamp`), ``twilight_temperature`` (`float`),
            ``dome_opening_3h`` (`float`) and ``morning_hvac_9h`` (`float`).
        """
        curve = self.predict(grid)
        twilight_time = self._twilight_time(curve)
        twilight_unix = twilight_time.timestamp()
        return {
            "curve": curve,
            "twilight_time": twilight_time,
            "twilight_temperature": self.predict_temperature_at_time(
                curve, twilight_unix
            ),
            "dome_opening_3h": self.predict_temperature_at_time(
                curve, twilight_unix - DOME_OPENING_LEAD_H * 3600.0
            ),
            "morning_hvac_9h": self.predict_temperature_at_time(
                curve, twilight_unix - MORNING_HVAC_LEAD_H * 3600.0
            ),
        }

    def forecast_window(self, grid: pd.DataFrame) -> pd.DataFrame:
        """Return the past observations plus the forward half-solar-day forecast.

        The forecast is valid for *any* moment of the day -- it is the
        continuous corrected temperature curve over the NBEATSx horizon
        (~half a solar day, ``phi`` -> ``phi + 0.5``), not a single twilight
        value.  Past rows carry the measured ``temp_actual`` (no forecast);
        forward rows carry the corrected ``forecast`` and its +/- band.

        Returns
        -------
        `pandas.DataFrame`
            Columns ``ds_real`` (UTC), ``SolarTime``, ``temp_actual``,
            ``forecast``, ``forecast_min`` and ``forecast_max``.  Suitable for
            assembly into the dashboard CSV by
            `~lsst.ts.weathernbeats.feed.build_feed_csv`.
        """
        curve = self.predict(grid)
        clean = grid.dropna(subset=["y"])

        past = pd.DataFrame(
            {
                "ds_real": pd.to_datetime(clean["ds_real"].to_numpy()),
                "SolarTime": clean["SolarTime"].to_numpy(),
                "temp_actual": clean["y"].to_numpy(),
                "forecast": np.nan,
                "forecast_min": np.nan,
                "forecast_max": np.nan,
            }
        )
        future = pd.DataFrame(
            {
                "ds_real": pd.to_datetime(curve["ds_real"].to_numpy()),
                "SolarTime": curve["SolarTime"].to_numpy(),
                "temp_actual": np.nan,
                "forecast": curve["T_forecast"].to_numpy(),
                "forecast_min": curve["T_forecast"].to_numpy() - curve["T_std"].to_numpy(),
                "forecast_max": curve["T_forecast"].to_numpy() + curve["T_std"].to_numpy(),
            }
        )
        return pd.concat([past, future], ignore_index=True)
