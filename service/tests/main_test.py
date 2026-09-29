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

"""Tests for the application lifespan in weathernbeats_service.main."""

from pathlib import Path

import pytest
from fastapi import FastAPI

from weathernbeats_service.config import config


@pytest.mark.asyncio
async def test_lifespan_loads_bundle_once(
    app: FastAPI, loaded_bundles: list[Path]
) -> None:
    """Test that startup loads the configured bundle, and only once."""
    assert loaded_bundles == [config.bundle]
