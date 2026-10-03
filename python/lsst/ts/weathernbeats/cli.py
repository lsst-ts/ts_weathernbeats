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

"""Command-line entry points for ts_weathernbeats."""

from __future__ import annotations

import argparse
import asyncio

from .feature_builder import FeatureBuilder
from .model import WeatherForecastModel
from .train import main as run_training

__all__ = ["run_forecast", "run_training"]


def run_forecast(argv: list[str] | None = None) -> None:
    """Run a single forecast: acquire telemetry, predict, print read-offs."""
    parser = argparse.ArgumentParser(description="Run a single NBEATSx+Ridge forecast.")
    parser.add_argument("--bundle", required=True, help="Model bundle directory.")
    parser.add_argument(
        "--simulation-mode",
        type=int,
        default=0,
        help="Non-zero uses the MockClient instead of a live EFD.",
    )
    args = parser.parse_args(argv)

    fb = FeatureBuilder(simulation_mode=args.simulation_mode)
    grid = asyncio.run(fb.get_features())
    model = WeatherForecastModel.load(args.bundle)
    result = model.operational_forecast(grid)

    print(f"twilight_time:        {result['twilight_time']}")
    print(f"twilight_temperature: {result['twilight_temperature']:.2f} C")
    print(f"dome_opening_3h:      {result['dome_opening_3h']:.2f} C")
    print(f"morning_hvac_9h:      {result['morning_hvac_9h']:.2f} C")
