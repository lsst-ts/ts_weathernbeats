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

"""The main application factory for the weathernbeats service.

Notes
-----
Be aware that, following the normal pattern for FastAPI services, the app is
constructed when this module is loaded and is not deferred until a function is
called.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from importlib.metadata import metadata, version

import structlog
from fastapi import FastAPI
from safir.fastapi import ClientRequestError, client_request_error_handler
from safir.logging import configure_logging, configure_uvicorn_logging
from safir.middleware.x_forwarded import XForwardedMiddleware
from safir.slack.webhook import SlackRouteErrorHandler

from .config import config
from .dependencies import forecast_service_dependency
from .handlers.external import external_router
from .handlers.internal import internal_router
from .services.forecast import ForecastService

__all__ = ["app"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Set up and tear down the application."""
    # Load the model bundle and connect to the EFD once, at startup, so that
    # a bad bundle path or EFD configuration fails the deployment.
    forecast_service_dependency.initialize(ForecastService.from_config(config))

    yield

    await forecast_service_dependency.aclose()


configure_logging(
    profile=config.log_profile,
    log_level=config.log_level,
    name="weathernbeats_service",
)
configure_uvicorn_logging(config.log_level)

app = FastAPI(
    title="weathernbeats",
    description=metadata("weathernbeats-service")["Summary"],
    version=version("weathernbeats-service"),
    openapi_url=f"{config.path_prefix}/openapi.json",
    docs_url=f"{config.path_prefix}/docs",
    redoc_url=f"{config.path_prefix}/redoc",
    lifespan=lifespan,
)
"""The main FastAPI application for weathernbeats."""

# Attach the routers.
app.include_router(internal_router)
app.include_router(external_router, prefix=f"{config.path_prefix}")

# Add middleware.
app.add_middleware(XForwardedMiddleware)

# Add error handlers.
app.exception_handler(ClientRequestError)(client_request_error_handler)

# Configure Slack alerts.
if config.slack_webhook:
    logger = structlog.get_logger("weathernbeats_service")
    SlackRouteErrorHandler.initialize(
        config.slack_webhook, "weathernbeats", logger
    )
    logger.debug("Initialized Slack webhook")
