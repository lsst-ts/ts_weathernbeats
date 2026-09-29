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

"""Models for weathernbeats."""

from datetime import datetime

from pydantic import BaseModel, Field
from safir.metadata import Metadata as SafirMetadata

__all__ = ["ForecastCurve", "ForecastPoint", "ForecastValue", "Index"]


class Index(BaseModel):
    """Metadata returned by the external root URL of the application."""

    metadata: SafirMetadata = Field(..., title="Package metadata")


class ForecastPoint(BaseModel):
    """Forecast temperature at one step of the forecast horizon."""

    time: datetime = Field(..., title="Time of the step (UTC)")

    temperature: float = Field(
        ..., title="Forecast temperature in degrees Celsius"
    )

    std: float = Field(
        ...,
        title="Forecast uncertainty in degrees Celsius",
        description="One-sigma band, growing with the forecast lead time",
    )


class ForecastCurve(BaseModel):
    """Calibrated temperature forecast over the whole horizon."""

    curve: list[ForecastPoint] = Field(
        ...,
        title="Forecast curve",
        description=(
            "One point per step of the solar-time grid, covering about half"
            " a solar day. Steps are uniform in solar time, so their"
            " wall-clock spacing is uneven: shorter by day, longer at night."
        ),
    )


class ForecastValue(BaseModel):
    """Forecast temperature interpolated at a requested time."""

    time: datetime = Field(..., title="Requested time (UTC)")

    temperature: float = Field(
        ..., title="Forecast temperature in degrees Celsius"
    )
