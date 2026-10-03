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

"""Handlers for the app's external root, ``/weathernbeats/``."""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from safir.metadata import get_metadata
from safir.models import ErrorLocation, ErrorModel
from safir.slack.webhook import SlackRouteErrorHandler

from ..config import config
from ..dependencies import forecast_service_dependency
from ..exceptions import TimeOutOfRangeError
from ..models import ForecastCurve, ForecastValue, Index
from ..services.forecast import ForecastService

__all__ = ["external_router"]

external_router = APIRouter(route_class=SlackRouteErrorHandler)
"""FastAPI router for all external handlers."""


@external_router.get(
    "/",
    description=(
        "Return metadata about the running application. The forecast itself"
        " is served by the forecast route under this prefix."
    ),
    response_model_exclude_none=True,
    summary="Application metadata",
)
async def get_index() -> Index:
    metadata = get_metadata(
        package_name="weathernbeats-service",
        application_name=config.name,
    )
    return Index(metadata=metadata)


@external_router.get(
    "/forecast",
    description=(
        "Forecast the temperature from the latest EFD telemetry. Without a"
        " time, return the calibrated curve over the whole horizon, about"
        " half a solar day ahead. With a time, return the temperature"
        " interpolated along that curve at that moment."
    ),
    responses={
        422: {
            "description": "Invalid time, or time outside the horizon",
            "model": ErrorModel,
        }
    },
    summary="Temperature forecast",
)
async def get_forecast(
    forecast_service: Annotated[
        ForecastService, Depends(forecast_service_dependency)
    ],
    time: Annotated[
        datetime | None,
        Query(
            title="Time to forecast",
            description=(
                "ISO 8601 time within the forecast horizon. Times without a"
                " timezone are taken to be UTC. Omit for the whole curve."
            ),
            examples=["2026-03-21T04:00:00Z"],
        ),
    ] = None,
) -> ForecastCurve | ForecastValue:
    if time is None:
        return await forecast_service.get_curve()
    try:
        return await forecast_service.get_value(time)
    except TimeOutOfRangeError as e:
        e.location = ErrorLocation.query
        e.field_path = ["time"]
        raise
