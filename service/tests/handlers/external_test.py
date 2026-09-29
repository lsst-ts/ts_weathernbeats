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

"""Tests for the weathernbeats_service.handlers.external module and routes."""

import math
from datetime import UTC, datetime
from itertools import pairwise

import pytest
from httpx import AsyncClient
from lsst.ts.weathernbeats import NBEATS_HORIZON

from weathernbeats_service.config import config


@pytest.mark.asyncio
async def test_get_index(client: AsyncClient) -> None:
    """Test ``GET /weathernbeats/``."""
    response = await client.get("/weathernbeats/")
    assert response.status_code == 200
    data = response.json()
    metadata = data["metadata"]
    assert metadata["name"] == config.name
    assert isinstance(metadata["version"], str)
    assert isinstance(metadata["description"], str)
    assert isinstance(metadata["repository_url"], str)
    assert isinstance(metadata["documentation_url"], str)


@pytest.mark.asyncio
async def test_get_forecast(client: AsyncClient) -> None:
    """Test ``GET /weathernbeats/forecast`` without a time."""
    response = await client.get("/weathernbeats/forecast")
    assert response.status_code == 200
    curve = response.json()["curve"]
    assert len(curve) == NBEATS_HORIZON

    times = [datetime.fromisoformat(p["time"]) for p in curve]
    assert all(t.tzinfo == UTC for t in times)
    assert all(a < b for a, b in pairwise(times))
    assert all(math.isfinite(p["temperature"]) for p in curve)
    assert all(p["std"] >= 0 for p in curve)


@pytest.mark.asyncio
async def test_get_forecast_at_time(client: AsyncClient) -> None:
    """Test ``GET /weathernbeats/forecast`` at a time in the horizon."""
    curve = (await client.get("/weathernbeats/forecast")).json()["curve"]
    time = curve[len(curve) // 2]["time"]

    response = await client.get(
        "/weathernbeats/forecast", params={"time": time}
    )
    assert response.status_code == 200
    data = response.json()
    assert datetime.fromisoformat(data["time"]) == datetime.fromisoformat(time)
    assert math.isfinite(data["temperature"])


@pytest.mark.asyncio
async def test_get_forecast_naive_time(client: AsyncClient) -> None:
    """Test that a time without a timezone is taken to be UTC."""
    curve = (await client.get("/weathernbeats/forecast")).json()["curve"]
    time = datetime.fromisoformat(curve[1]["time"])
    naive = time.replace(tzinfo=None).isoformat()

    response = await client.get(
        "/weathernbeats/forecast", params={"time": naive}
    )
    assert response.status_code == 200
    assert datetime.fromisoformat(response.json()["time"]) == time


@pytest.mark.asyncio
async def test_get_forecast_invalid_time(client: AsyncClient) -> None:
    """Test ``GET /weathernbeats/forecast`` with an unparseable time."""
    response = await client.get(
        "/weathernbeats/forecast", params={"time": "not-a-time"}
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["query", "time"]


@pytest.mark.asyncio
async def test_get_forecast_time_out_of_range(client: AsyncClient) -> None:
    """Test ``GET /weathernbeats/forecast`` at a time outside the horizon."""
    response = await client.get(
        "/weathernbeats/forecast", params={"time": "2000-01-01T00:00:00Z"}
    )
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["type"] == "time_out_of_range"
    assert error["loc"] == ["query", "time"]
    assert "outside the forecast horizon" in error["msg"]
