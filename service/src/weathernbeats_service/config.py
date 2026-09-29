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

"""Configuration definition."""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from safir.logging import LogLevel, Profile

__all__ = ["Config", "config"]


class Config(BaseSettings):
    """Configuration for weathernbeats."""

    model_config = SettingsConfigDict(
        env_prefix="WEATHERNBEATS_", case_sensitive=False
    )

    bundle: Path = Field(
        title="Model bundle directory",
        description=(
            "Directory holding a trained NBEATSx+Ridge model bundle, as"
            " written by train_weathernbeats"
        ),
    )

    simulation: bool = Field(
        False,
        title="Use simulated telemetry",
        description=(
            "If true, forecast from the synthetic telemetry of the"
            " ts_weathernbeats MockClient instead of querying the EFD"
            " selected by LSST_SITE"
        ),
    )

    log_level: LogLevel = Field(
        LogLevel.INFO, title="Log level of the application's logger"
    )

    log_profile: Profile = Field(
        Profile.development, title="Application logging profile"
    )

    name: str = Field("weathernbeats", title="Name of application")

    path_prefix: str = Field(
        "/weathernbeats", title="URL prefix for application"
    )

    slack_webhook: SecretStr | None = Field(
        None,
        title="Slack webhook for alerts",
        description="If set, alerts will be posted to this Slack webhook",
    )


config = Config()
"""Configuration for weathernbeats."""
