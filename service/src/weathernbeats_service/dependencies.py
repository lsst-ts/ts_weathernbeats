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

"""Dependencies for the weathernbeats service."""

from .services.forecast import ForecastService

__all__ = ["ForecastServiceDependency", "forecast_service_dependency"]


class ForecastServiceDependency:
    """Provide the forecast service created at application startup.

    Loading the model bundle and connecting to the EFD are too slow to do on
    every request, so the application lifespan creates the service once and
    this dependency hands it to the request handlers.
    """

    def __init__(self) -> None:
        self._service: ForecastService | None = None

    async def __call__(self) -> ForecastService:
        if self._service is None:
            raise RuntimeError("ForecastServiceDependency not initialized")
        return self._service

    def initialize(self, service: ForecastService) -> None:
        """Set the forecast service returned by the dependency.

        Parameters
        ----------
        service
            Forecast service to provide to request handlers.
        """
        self._service = service

    async def aclose(self) -> None:
        """Close the forecast service, if one was set."""
        if self._service is not None:
            await self._service.aclose()
            self._service = None


forecast_service_dependency = ForecastServiceDependency()
"""The dependency that provides the forecast service."""
