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

"""Service that forecasts the temperature from the latest EFD telemetry."""

import asyncio
import math
from datetime import UTC, datetime
from typing import Self

import pandas as pd
from lsst.ts.weathernbeats import FeatureBuilder, WeatherForecastModel

from ..config import Config
from ..exceptions import TimeOutOfRangeError
from ..models import ForecastCurve, ForecastPoint, ForecastValue

__all__ = ["ForecastService"]


class ForecastService:
    """Forecast the temperature from the latest EFD telemetry.

    Parameters
    ----------
    model
        Loaded two-stage NBEATSx+Ridge model.
    feature_builder
        Feature builder whose EFD client has already been created.
    """

    def __init__(
        self, model: WeatherForecastModel, feature_builder: FeatureBuilder
    ) -> None:
        self._model = model
        self._feature_builder = feature_builder

        # NeuralForecast attaches a new Lightning trainer to the shared model
        # on every prediction, so only run one prediction at a time.
        self._lock = asyncio.Lock()

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """Load the model bundle and connect to the EFD.

        Parameters
        ----------
        config
            Application configuration.

        Returns
        -------
        ForecastService
            Service ready to produce forecasts.
        """
        model = WeatherForecastModel.load(config.bundle)
        feature_builder = FeatureBuilder(
            simulation_mode=int(config.simulation)
        )
        feature_builder.create_client()
        return cls(model, feature_builder)

    async def aclose(self) -> None:
        """Close the connection to the EFD."""
        # EfdClient has no close method, but the InfluxDB client it wraps
        # does. The simulation MockClient has nothing to close.
        influx_client = getattr(
            self._feature_builder.client, "_influx_client", None
        )
        close = getattr(influx_client, "close", None)
        if close:
            await close()

    async def get_curve(self) -> ForecastCurve:
        """Forecast the temperature over the whole horizon.

        Returns
        -------
        ForecastCurve
            Calibrated temperature and uncertainty at every horizon step.
        """
        curve = await self._predict()
        points = [
            ForecastPoint(time=time, temperature=temperature, std=std)
            for time, temperature, std in zip(
                _utc_times(curve),
                curve["T_forecast"],
                curve["T_std"],
                strict=True,
            )
        ]
        return ForecastCurve(curve=points)

    async def get_value(self, time: datetime) -> ForecastValue:
        """Forecast the temperature at one moment in the horizon.

        Parameters
        ----------
        time
            Time to forecast. Times without a timezone are taken to be UTC.

        Returns
        -------
        ForecastValue
            Temperature interpolated along the forecast curve at ``time``.

        Raises
        ------
        TimeOutOfRangeError
            Raised if ``time`` is outside the forecast horizon.
        """
        if time.tzinfo is None:
            time = time.replace(tzinfo=UTC)
        time = time.astimezone(UTC)
        curve = await self._predict()
        temperature = self._model.predict_temperature_at_time(
            curve, pd.Timestamp(time)
        )
        if math.isnan(temperature):
            times = _utc_times(curve)
            msg = (
                f"{time.isoformat()} is outside the forecast horizon,"
                f" {times[0].isoformat()} to {times[-1].isoformat()}"
            )
            raise TimeOutOfRangeError(msg)
        return ForecastValue(time=time, temperature=temperature)

    async def _predict(self) -> pd.DataFrame:
        """Query the EFD and forecast from the latest telemetry."""
        async with self._lock:
            telemetry = await self._feature_builder.query()
            return await asyncio.to_thread(self._predict_sync, telemetry)

    def _predict_sync(self, telemetry: pd.DataFrame) -> pd.DataFrame:
        """Build the solar-grid features and run the model (CPU-bound)."""
        frame = FeatureBuilder.setup_fit(telemetry)
        return self._model.predict(FeatureBuilder.build_grid(frame))


def _utc_times(curve: pd.DataFrame) -> list[datetime]:
    """Return the real-time column of a forecast curve as UTC datetimes."""
    times = pd.to_datetime(curve["ds_real"]).dt.round("us").dt.tz_localize(UTC)
    return [time.to_pydatetime() for time in times]
