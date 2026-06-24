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

"""FeatureBuilder: acquire temperature telemetry from the EFD and build the
solar-grid feature frame consumed by the two-stage forecaster.

Data acquisition mirrors ``ts_weatherforecast``'s ``BobDobbs`` model: the EFD
client is selected from the ``LSST_SITE`` environment variable, with a
``MockClient`` fallback so ``simulation_mode`` works without a live EFD.  The
runtime query pulls the 1-minute mean of ``temperatureItem0`` from
``lsst.sal.ESS.temperature`` (salIndex 301, the Summit Weather Tower sensor)
over a 7-day window, then layers the solar-grid feature engineering on top.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import pandas as pd
from lsst_efd_client import EfdClient

from . import solar_grid

__all__ = [
    "MockClient",
    "FeatureBuilder",
    "EFD_SITES",
    "NBEATS_INPUT_SIZE",
    "NBEATS_HORIZON",
    "NB_HIST",
    "NB_FUTR",
    "RIDGE_FEATS",
    "SPREAD_THRESHOLD",
]

# EFD instance per LSST_SITE (mirrors ts_weatherforecast.utils.efd_sites).
EFD_SITES = {
    "summit": "summit_efd",
    "base": "base_efd",
    "usdf": "usdf_efd",
    "tucson": "tucson_teststand_efd",
    "weatherforecast": None,
    "test": None,
}

# NBEATSx geometry (must match the trained artifacts; see config in retrain).
NBEATS_INPUT_SIZE = 48  # 1 solar-day lookback
NBEATS_HORIZON = 26     # covers up to 12 h lead

# High intra-interval spread (deg C) above which a raw point is dropped and
# causally forward-filled (no future leakage).
SPREAD_THRESHOLD = 3.0

# NBEATSx exogenous features (direct-T formulation, from the reference sweep).
NB_HIST = ["y_raw", "y_lag_24", "trend_solar_2h", "y_diff_24h"]
NB_FUTR = ["solar_sin", "solar_cos", "doy_sin", "doy_cos"]

# Per-slot Ridge features (v1: temperature + solar-time only; humidity/wind
# auxiliary inputs are omitted in v1).  The NBEATSx prediction is appended as
# the final feature at fit/predict time.
RIDGE_FEATS = [
    "y_raw", "y_lag_6", "y_lag_12", "y_lag_24", "y_lag_48",
    "trend_solar_2h", "trend_solar_4h", "solar_sin", "solar_cos",
    "doy_sin", "doy_cos", "dmean_1d",
]


def _backward_ols_slope(y: pd.Series, window: int) -> pd.Series:
    """Vectorized backward OLS slope (per step) over a rolling ``window``."""
    idx = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    y_mean = y.rolling(window, min_periods=window).mean()
    x_mean = idx.rolling(window, min_periods=window).mean()
    xy_mean = (y * idx).rolling(window, min_periods=window).mean()
    x2_mean = (idx ** 2).rolling(window, min_periods=window).mean()
    cov_xy = xy_mean - x_mean * y_mean
    var_x = x2_mean - x_mean ** 2
    return cov_xy / var_x


class MockClient:
    """A mock EFD client returning synthetic diurnal temperature telemetry."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def query(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Return a fake 7-day, 1-minute-cadence temperature series."""
        n = 7 * 24 * 60
        end = pd.Timestamp("2026-03-21T00:00:00")  # near an equinox
        index = pd.date_range(end=end, periods=n, freq="min")
        hours = index.hour + index.minute / 60.0
        # Simple diurnal cycle + small noise; deterministic-ish via stdlib.
        diurnal = 10.0 - 6.0 * np.cos(2 * np.pi * (hours - 15.0) / 24.0)
        noise = np.array([random.uniform(-0.3, 0.3) for _ in range(n)])
        return pd.DataFrame({"mean_temperature": diurnal + noise}, index=index)


class FeatureBuilder:
    """Acquire EFD telemetry and build the solar-grid feature frame.

    Parameters
    ----------
    simulation_mode : `int`, optional
        When non-zero (or when ``LSST_SITE`` is not a live EFD), the
        `MockClient` is used so the pipeline runs without a live EFD.
    """

    def __init__(self, simulation_mode: int = 0) -> None:
        self.simulation_mode = simulation_mode
        self.client: EfdClient | None = None

    def create_client(self) -> None:
        """Create the EFD client, selecting the instance from ``LSST_SITE``.

        A live `~lsst_efd_client.EfdClient` is created whenever ``LSST_SITE``
        maps to a real EFD instance and ``simulation_mode`` is off; otherwise
        the `MockClient` fallback is used so the pipeline runs without an EFD.
        """
        efd_uri = EFD_SITES.get(os.getenv("LSST_SITE", "test"))
        if not self.simulation_mode and efd_uri is not None:
            self.client = EfdClient(efd_uri)
        else:
            self.client = EfdClient("summit_efd_copy", client=MockClient())

    async def query(self) -> pd.DataFrame:
        """Return the recent temperature history from the EFD.

        Reuses ``ts_weatherforecast``'s query: the 1-minute mean of
        ``temperatureItem0`` from ``lsst.sal.ESS.temperature`` (salIndex 301)
        over a 7-day window with linear gap fill.
        """
        assert self.client is not None
        query = " ".join(
            (
                "SELECT mean(temperatureItem0) as mean_temperature FROM",
                '"efd"."autogen"."lsst.sal.ESS.temperature"',
                "where salIndex=301 AND time > now() - 7d GROUP BY time(1m) FILL(linear)",
            )
        )
        if hasattr(self.client, "_influx_client"):
            return await self.client._influx_client.query(query)
        return await self.client.influx_client.query(query)

    @staticmethod
    def setup_fit(results: pd.DataFrame) -> pd.DataFrame:
        """Reshape the EFD response into a tz-naive ``ds``/``y`` frame.

        Drops high-intra-interval-spread points (when ``max``/``min`` columns
        are present) and causally forward-fills the gaps.
        """
        results = results.copy()
        results.insert(0, "ds", pd.to_datetime(results.index))
        results.index = range(results.shape[0])
        results["ds"] = pd.to_datetime(results["ds"], utc=True).dt.tz_localize(None)
        results = results.rename(columns={"mean_temperature": "y"})

        if {"max", "min"}.issubset(results.columns):
            spread = results["max"] - results["min"]
            results.loc[spread > SPREAD_THRESHOLD, "y"] = np.nan
        results["y"] = results["y"].ffill().bfill()  # causal: no future leak
        return results[["ds", "y"]].dropna().drop_duplicates("ds").sort_values("ds")

    @staticmethod
    def build_grid(df: pd.DataFrame) -> pd.DataFrame:
        """Resample a ``ds``/``y`` frame onto the solar grid and add features.

        Runs sun-altitude/event detection, solar-time coordinates and
        resampling onto the 48-step solar grid, then adds the lag, trend,
        difference and phase features required by both forecaster stages.
        """
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"], utc=True).dt.tz_localize(None)
        df["alt_sun"] = solar_grid.get_sun_altitude(pd.DatetimeIndex(df["ds"]))
        df = solar_grid.detect_events(df)
        df = solar_grid.add_solar_time(df)
        grid = solar_grid.resample_to_solar_grid(df)

        y = grid["y"]
        grid["y_raw"] = y
        grid["y_lag_6"] = y.shift(6)
        grid["y_lag_12"] = y.shift(12)
        grid["y_lag_24"] = y.shift(24)
        grid["y_lag_48"] = y.shift(48)
        grid["y_diff_24h"] = y - y.shift(48)
        grid["trend_solar_2h"] = _backward_ols_slope(y, 4)
        grid["trend_solar_4h"] = _backward_ols_slope(y, 8)

        mean_24h = y.rolling(48, min_periods=24).mean()
        grid["dmean_1d"] = mean_24h - mean_24h.shift(48)

        grid["solar_sin"] = np.sin(2 * np.pi * grid["SolarTime"])
        grid["solar_cos"] = np.cos(2 * np.pi * grid["SolarTime"])
        doy = grid["ds_real"].dt.dayofyear
        grid["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        grid["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        return grid

    async def get_features(self) -> pd.DataFrame:
        """End-to-end: create client, query the EFD, build the feature grid."""
        if self.client is None:
            self.create_client()
        results = await self.query()
        frame = self.setup_fit(results)
        return self.build_grid(frame)
