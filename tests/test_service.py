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

"""Tests for the FastAPI forecast service."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lsst.ts.weathernbeats.model import WeatherForecastModel
from lsst.ts.weathernbeats.service import app, get_model


@pytest.fixture
def client(fake_bundle, monkeypatch):
    """A TestClient with the model dependency swapped for the fake bundle."""
    monkeypatch.setenv("WEATHERNBEATS_SIMULATION", "1")
    app.dependency_overrides[get_model] = lambda: WeatherForecastModel(fake_bundle)
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_forecast_returns_full_curve(client):
    resp = client.get("/forecast")
    assert resp.status_code == 200
    curve = resp.json()["curve"]
    assert len(curve) > 0
    assert all(np.isfinite(row["temperature"]) for row in curve)


def test_forecast_interpolates_at_time(client):
    curve = client.get("/forecast").json()["curve"]
    mid_time = curve[len(curve) // 2]["time"]
    resp = client.get("/forecast", params={"time": mid_time})
    assert resp.status_code == 200
    assert np.isfinite(resp.json()["temperature"])


def test_forecast_rejects_bad_time(client):
    resp = client.get("/forecast", params={"time": "not-a-time"})
    assert resp.status_code == 400
