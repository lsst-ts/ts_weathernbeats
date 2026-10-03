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

"""Tests for the EFD FeatureBuilder path (using the MockClient).

The async EFD methods are driven via ``asyncio.run`` so the suite needs no
pytest async plugin.
"""

import asyncio

import pandas as pd

from lsst.ts.weathernbeats.feature_builder import (
    NB_FUTR,
    NB_HIST,
    RIDGE_FEATS,
    FeatureBuilder,
    MockClient,
)


def test_mock_client_query_shape():
    df = asyncio.run(MockClient().query())
    assert "mean_temperature" in df.columns
    assert len(df) == 7 * 24 * 60


def test_create_client_simulation_mode():
    fb = FeatureBuilder(simulation_mode=1)
    fb.create_client()
    assert fb.client is not None


def test_get_features_builds_all_columns():
    fb = FeatureBuilder(simulation_mode=1)
    grid = asyncio.run(fb.get_features())
    needed = set(NB_HIST) | set(NB_FUTR) | set(RIDGE_FEATS) | {"SolarTime", "ds", "ds_real"}
    assert needed.issubset(grid.columns)
    assert grid["solar_sin"].between(-1, 1).all()
    assert grid["solar_cos"].between(-1, 1).all()


def test_setup_fit_returns_naive_ds():
    fb = FeatureBuilder(simulation_mode=1)
    fb.create_client()
    results = asyncio.run(fb.query())
    frame = fb.setup_fit(results)
    assert {"ds", "y"} == set(frame.columns)
    assert pd.api.types.is_datetime64_ns_dtype(frame["ds"])
    assert frame["ds"].dt.tz is None
