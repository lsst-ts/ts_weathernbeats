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

"""Shared pytest fixtures for ts_weathernbeats tests."""

import asyncio

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from lsst.ts.weathernbeats.artifacts import ModelBundle, SlotModel
from lsst.ts.weathernbeats.feature_builder import (
    NBEATS_HORIZON,
    RIDGE_FEATS,
    FeatureBuilder,
)
from lsst.ts.weathernbeats.solar_grid import STEPS_PER_DAY
from lsst.ts.weathernbeats.solar_slots import (
    phi_to_equinox_label,
    solar_slots,
)


class FakeNeuralForecast:
    """A stand-in for NeuralForecast that returns a deterministic trajectory.

    Lets the two-stage inference logic be tested without training a real
    NBEATSx network (which would pull in PyTorch Lightning).  The "forecast"
    simply persists the last observed temperature across the horizon.
    """

    def predict(self, df, futr_df):
        out = futr_df[["unique_id", "ds"]].copy()
        last_y = float(df["y"].iloc[-1])
        out["NBEATSx"] = last_y
        return out

    def save(self, *args, **kwargs):  # pragma: no cover - not used in unit tests
        raise NotImplementedError("FakeNeuralForecast cannot be persisted.")


@pytest.fixture
def synthetic_grid():
    """A solar-grid feature frame built from the MockClient telemetry."""
    fb = FeatureBuilder(simulation_mode=1)
    return asyncio.run(fb.get_features())


@pytest.fixture
def fake_bundle(synthetic_grid):
    """A ModelBundle with a FakeNeuralForecast and real per-slot Ridge models.

    Each slot's Ridge is fit on a tiny synthetic regression so inference runs
    end to end; the point is to exercise selection + two-stage logic, not
    forecast accuracy.
    """
    feat_cols = [c for c in RIDGE_FEATS if c in synthetic_grid.columns]
    clean = synthetic_grid.dropna(subset=feat_cols + ["y"]).reset_index(drop=True)
    x = clean[feat_cols].to_numpy(dtype=float)
    # Append a synthetic NBEATSx-prediction column (= y, the persistence proxy).
    x = np.column_stack([x, clean["y"].to_numpy(dtype=float)])
    y = clean["y"].to_numpy(dtype=float)

    scaler = StandardScaler().fit(x)
    ridge = Ridge(alpha=1.0).fit(scaler.transform(x), y)

    slots = {}
    for phi in solar_slots(STEPS_PER_DAY):
        phi = float(phi)
        slots[phi] = SlotModel(
            phi=phi,
            equinox_label=phi_to_equinox_label(phi),
            feature_list=feat_cols,
            scaler=scaler,
            ridge=ridge,
        )
    metadata = {
        "model_version": "test",
        "nbeats_horizon": NBEATS_HORIZON,
        "ridge_features": feat_cols,
    }
    return ModelBundle(nbeatsx=FakeNeuralForecast(), slots=slots, metadata=metadata)
