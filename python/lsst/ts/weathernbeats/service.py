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

"""FastAPI service exposing the NBEATSx+Ridge forecast over HTTP.

Lets ``ts_weatherforecast`` (or anything else) fetch predicted temperatures
via a single API call instead of importing this package and running the CLI.
The model bundle is loaded once (from ``WEATHERNBEATS_BUNDLE``) and reused
across requests; the EFD telemetry is re-queried on every request so the
forecast reflects the latest data.
"""

from __future__ import annotations

import os

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query

from .feature_builder import FeatureBuilder
from .model import WeatherForecastModel

__all__ = ["app", "get_model", "serve"]

app = FastAPI(title="ts_weathernbeats forecast service")

_model: WeatherForecastModel | None = None


def get_model() -> WeatherForecastModel:
    """Load (once) and return the model bundle from ``WEATHERNBEATS_BUNDLE``."""
    global _model
    if _model is None:
        _model = WeatherForecastModel.load(os.environ["WEATHERNBEATS_BUNDLE"])
    return _model


@app.get("/forecast")
async def forecast(
    time: str | None = Query(
        None, description="ISO8601 UTC timestamp; omit for the full curve."
    ),
    model: WeatherForecastModel = Depends(get_model),
) -> dict:
    """Return the calibrated forecast curve, or one value interpolated at ``time``.

    Without ``time``, returns every horizon step (``time``, ``temperature``,
    ``std``). With ``time``, returns a single interpolated ``temperature`` at
    that instant -- the same read-off ``operational_forecast`` uses
    internally for twilight/dome/HVAC setpoints, but for any moment.
    """
    simulation_mode = int(os.getenv("WEATHERNBEATS_SIMULATION", "0"))
    grid = await FeatureBuilder(simulation_mode=simulation_mode).get_features()
    curve = model.predict(grid)

    if time is not None:
        try:
            target = pd.Timestamp(time)
        except ValueError:
            raise HTTPException(400, f"Bad time value: {time!r}")
        value = model.predict_temperature_at_time(curve, target)
        return {"time": target.isoformat(), "temperature": value}

    return {
        "curve": [
            {
                "time": pd.Timestamp(row.ds_real).isoformat(),
                "temperature": row.T_forecast,
                "std": row.T_std,
            }
            for row in curve.itertuples()
        ]
    }


def serve() -> None:
    """Entry point (``serve_weathernbeats``): run the service with uvicorn."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
