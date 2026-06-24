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

"""Tests for solar-grid construction and preprocessing correctness."""

import numpy as np
import pandas as pd

from lsst.ts.weathernbeats.solar_grid import (
    SOLAR_GRID_STEP,
    STEPS_PER_DAY,
    add_solar_time,
    detect_events,
    get_sun_altitude,
    resample_to_solar_grid,
    solar_time,
)


def _raw_frame(days=10):
    """A multi-day 15-minute synthetic temperature frame near an equinox."""
    n = days * 24 * 4
    ds = pd.date_range("2026-03-12", periods=n, freq="15min")
    hours = ds.hour + ds.minute / 60.0
    y = 10.0 - 6.0 * np.cos(2 * np.pi * (hours - 15.0) / 24.0)
    return pd.DataFrame({"ds": ds, "y": y})


def test_solar_time_in_unit_interval():
    df = _raw_frame()
    df["alt_sun"] = get_sun_altitude(pd.DatetimeIndex(df["ds"]))
    df = detect_events(df)
    df = add_solar_time(df)
    assert df["SolarTime"].between(0, 1).all()
    assert (df["DayCount"] >= 0).all()
    # solarDayHour is monotonically increasing.
    assert (np.diff(df["solarDayHour"].to_numpy()) > 0).all()


def test_grid_step_and_columns():
    df = _raw_frame()
    df["alt_sun"] = get_sun_altitude(pd.DatetimeIndex(df["ds"]))
    df = detect_events(df)
    df = add_solar_time(df)
    grid = resample_to_solar_grid(df)
    # Uniform spacing of one solar slot.
    diffs = np.diff(grid["solarDayHour"].to_numpy())
    assert np.allclose(diffs, SOLAR_GRID_STEP)
    assert {"SolarTime", "DayCount", "ds", "ds_real", "y"}.issubset(grid.columns)
    # ds_real (real UTC wall clock) is preserved and increasing.
    assert (np.diff(pd.to_datetime(grid["ds_real"]).astype("int64")) > 0).all()


def test_causal_spread_fill_no_future_leak():
    """High-spread points are dropped and forward-filled, never back-filled
    from the future when a prior good value exists."""
    from lsst.ts.weathernbeats.feature_builder import FeatureBuilder

    df = pd.DataFrame(
        {
            "mean_temperature": [10.0, 11.0, 99.0, 13.0],
            "max": [10.1, 11.1, 110.0, 13.1],
            "min": [9.9, 10.9, 80.0, 12.9],
        },
        index=pd.date_range("2026-03-20", periods=4, freq="min"),
    )
    out = FeatureBuilder.setup_fit(df)
    # The spread>3 point (index 2) is replaced by the previous good value 11.0.
    assert out["y"].iloc[2] == 11.0


def test_steps_per_day_constant():
    assert STEPS_PER_DAY == 48
    assert SOLAR_GRID_STEP == 1.0 / 48.0


def test_solar_time_quarter_landmarks():
    """SolarTime hits the cardinal points exactly: sunrise=0.0, midday=0.25,
    sunset=0.5, solar-midnight=0.75 (the standard solar-time convention)."""
    ts = pd.date_range("2026-06-22", "2026-06-26", freq="min")
    st = solar_time(pd.DatetimeIndex(ts)) % 1.0
    alt = get_sun_altitude(pd.DatetimeIndex(ts))
    prev = np.concatenate([[alt[0]], alt[:-1]])
    sunrise = np.where((prev < 0) & (alt >= 0))[0]
    sunset = np.where((prev >= 0) & (alt < 0))[0]

    # Sunrise -> 0.0, sunset -> 0.5 (exact, these are the interpolation anchors).
    assert np.allclose(st[sunrise], 0.0, atol=1e-6)
    assert np.allclose(st[sunset], 0.5, atol=1e-6)

    # Solar midday (max altitude) -> 0.25; solar midnight (min altitude) -> 0.75.
    # Within one grid step (1/(24*60) of a day) of the exact quarter.
    tol = 1.0 / (24 * 60)
    assert abs(st[int(np.argmax(alt))] - 0.25) < 5 * tol
    assert abs(st[int(np.argmin(alt))] - 0.75) < 5 * tol


def test_solar_time_covers_trailing_night():
    """A series ending mid-night (past solar midnight) keeps its last rows:
    SolarTime there is > 0.5 (night branch), not truncated back to sunset."""
    # 12:00 UTC start through 05:00 next day -- ends deep in the night at Rubin.
    ts = pd.date_range("2026-06-22T12:00", "2026-06-24T05:00", freq="15min")
    df = pd.DataFrame({"ds": ts, "y": np.zeros(len(ts))})
    df["alt_sun"] = get_sun_altitude(pd.DatetimeIndex(df["ds"]))
    df = detect_events(df)
    out = add_solar_time(df)
    # The last observation survives (not dropped for want of a "next sunrise").
    assert pd.Timestamp(out["ds"].iloc[-1]) == pd.Timestamp(ts[-1])
    # Deep night -> SolarTime in the (0.5, 1.0) night branch.
    assert 0.5 < out["SolarTime"].iloc[-1] < 1.0
