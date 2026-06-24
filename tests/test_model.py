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

"""Tests for the two-stage inference path and operational read-offs."""

import numpy as np
import pandas as pd
import pytest

from lsst.ts.weathernbeats.feature_builder import NBEATS_HORIZON
from lsst.ts.weathernbeats.model import WeatherForecastModel, interp_at


def test_interp_at_matches_notebook_semantics():
    """interp_at reproduces the notebook helper exactly."""
    temps = np.array([10.0, 12.0, 14.0, 16.0])
    snd = 1000.0
    delta = 300.0
    # index_float = (target - snd)/delta - 1; target one full step past snd+1step.
    target = snd + 2 * delta  # index_float = 1 -> temps[1] = 12
    assert interp_at(temps, snd, target, delta) == pytest.approx(12.0)
    # Out of range -> NaN.
    assert np.isnan(interp_at(temps, snd, snd, delta))


def test_predict_returns_horizon_curve(fake_bundle, synthetic_grid):
    model = WeatherForecastModel(fake_bundle)
    curve = model.predict(synthetic_grid)
    assert len(curve) == NBEATS_HORIZON
    assert {"ds", "ds_real", "SolarTime", "T_nb", "slot_phi", "T_forecast"}.issubset(
        curve.columns
    )
    assert curve["T_forecast"].notna().all()
    # Slot phi assigned per step is one of the trained slots.
    assert set(curve["slot_phi"]).issubset(set(fake_bundle.slots))


def test_predict_temperature_at_time_interpolates(fake_bundle, synthetic_grid):
    model = WeatherForecastModel(fake_bundle)
    curve = model.predict(synthetic_grid)
    mid = pd.Timestamp(curve["ds_real"].iloc[len(curve) // 2])
    val = model.predict_temperature_at_time(curve, mid)
    assert np.isfinite(val)
    # Outside the horizon -> NaN.
    after = pd.Timestamp(curve["ds_real"].iloc[-1]) + pd.Timedelta(days=2)
    assert np.isnan(model.predict_temperature_at_time(curve, after))


def test_operational_forecast_readoffs(fake_bundle, synthetic_grid):
    model = WeatherForecastModel(fake_bundle)
    result = model.operational_forecast(synthetic_grid)
    assert {"curve", "twilight_time", "twilight_temperature",
            "dome_opening_3h", "morning_hvac_9h"}.issubset(result)
    assert np.isfinite(result["twilight_temperature"])


def test_horizon_advances_half_solar_day(fake_bundle, synthetic_grid):
    """The forecast covers NBEATS_HORIZON solar steps: the last forecast solar
    position is phi_now + NBEATS_HORIZON/STEPS_PER_DAY (~phi_now + 0.54).

    Issued near sunrise (phi_now ~ 0), the curve therefore runs past sunset
    (0.5) -- i.e. it spans a little more than half a solar day."""
    from lsst.ts.weathernbeats.solar_grid import STEPS_PER_DAY

    model = WeatherForecastModel(fake_bundle)
    curve = model.predict(synthetic_grid)

    phi_now = float(synthetic_grid.dropna(subset=["y"])["SolarTime"].iloc[-1])
    advance = NBEATS_HORIZON / STEPS_PER_DAY
    expected_last = (phi_now + advance) % 1.0
    got_last = float(curve["SolarTime"].iloc[-1])
    assert got_last == pytest.approx(expected_last, abs=1e-6)
    # ~0.54 of a solar day, i.e. past the half-day (sunset) mark.
    assert advance > 0.5


def test_no_fit_at_inference(fake_bundle, synthetic_grid, monkeypatch):
    """Inference must never call .fit on scaler or ridge."""
    for slot in fake_bundle.slots.values():
        def _boom(*a, **k):
            raise AssertionError("fit called during inference")

        monkeypatch.setattr(slot.scaler, "fit", _boom, raising=False)
        monkeypatch.setattr(slot.ridge, "fit", _boom, raising=False)
    model = WeatherForecastModel(fake_bundle)
    model.predict(synthetic_grid)  # must not raise
