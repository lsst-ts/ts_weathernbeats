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

"""Test fixtures for weathernbeats service tests."""

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

import pandas as pd
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from lsst.ts.weathernbeats import (
    NBEATS_HORIZON,
    RIDGE_FEATS,
    STEPS_PER_DAY,
    FeatureBuilder,
    MockClient,
    ModelBundle,
    SlotModel,
    WeatherForecastModel,
    phi_to_equinox_label,
    solar_slots,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from weathernbeats_service import main


class FakeNeuralForecast:
    """Stand-in for NeuralForecast that persists the last observation.

    Lets the service run the real two-stage inference path without training
    an NBEATSx network.
    """

    def predict(self, df: pd.DataFrame, futr_df: pd.DataFrame) -> pd.DataFrame:
        forecast = futr_df[["unique_id", "ds"]].copy()
        forecast["NBEATSx"] = float(df["y"].iloc[-1])
        return forecast


class FakeEfdClient:
    """Stand-in for EfdClient that serves synthetic telemetry.

    EfdClient fetches credentials over the network even when given a mock
    InfluxDB client, so the tests replace it entirely.
    """

    def __init__(self) -> None:
        self._influx_client = MockClient()


@pytest.fixture(scope="session")
def fake_model() -> WeatherForecastModel:
    """Return a model with a fake NBEATSx stage and fitted Ridge slots."""
    telemetry = asyncio.run(MockClient().query())
    grid = FeatureBuilder.build_grid(FeatureBuilder.setup_fit(telemetry))
    features = [c for c in RIDGE_FEATS if c in grid.columns]
    clean = grid.dropna(subset=[*features, "y"])

    # The NBEATSx prediction is the last Ridge input; y stands in for it.
    x = clean[[*features, "y"]].to_numpy(dtype=float)
    scaler = StandardScaler().fit(x)
    ridge = Ridge(alpha=1.0).fit(
        scaler.transform(x), clean["y"].to_numpy(dtype=float)
    )
    slots = {
        float(phi): SlotModel(
            phi=float(phi),
            equinox_label=phi_to_equinox_label(float(phi)),
            feature_list=features,
            scaler=scaler,
            ridge=ridge,
        )
        for phi in solar_slots(STEPS_PER_DAY)
    }
    bundle = ModelBundle(
        nbeatsx=FakeNeuralForecast(),
        slots=slots,
        metadata={"model_version": "test", "nbeats_horizon": NBEATS_HORIZON},
    )
    return WeatherForecastModel(bundle)


@pytest.fixture
def loaded_bundles(
    fake_model: WeatherForecastModel, monkeypatch: pytest.MonkeyPatch
) -> list[Path]:
    """Replace model loading and the EFD client with fakes.

    Returns
    -------
    list of pathlib.Path
        Bundle paths the application has asked to load.
    """
    loaded: list[Path] = []

    def load(path: str | Path) -> WeatherForecastModel:
        loaded.append(Path(path))
        return fake_model

    def create_client(self: FeatureBuilder) -> None:
        self.client = FakeEfdClient()

    monkeypatch.setattr(WeatherForecastModel, "load", load)
    monkeypatch.setattr(FeatureBuilder, "create_client", create_client)
    return loaded


@pytest_asyncio.fixture
async def app(loaded_bundles: list[Path]) -> AsyncGenerator[FastAPI]:
    """Return a configured test application.

    Wraps the application in a lifespan manager so that startup and shutdown
    events are sent during test execution.
    """
    async with LifespanManager(main.app):
        yield main.app


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient]:
    """Return an ``httpx.AsyncClient`` configured to talk to the test app."""
    async with AsyncClient(
        base_url="https://example.com/", transport=ASGITransport(app=app)
    ) as client:
        yield client
