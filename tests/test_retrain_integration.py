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

"""End-to-end retrain -> save -> load -> inference on a small synthetic grid.

This exercises the real NBEATSx network and is slow, so it is opt-in via the
``WEATHERNBEATS_SLOW`` environment variable.  It guards the artifact contract:
a freshly trained bundle reloads and produces a finite continuous forecast.
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("WEATHERNBEATS_SLOW"),
    reason="slow NBEATSx training; set WEATHERNBEATS_SLOW=1 to run",
)


def test_retrain_and_infer(tmp_path):
    from lsst.ts.weathernbeats import train as train_mod
    from lsst.ts.weathernbeats.artifacts import ModelBundle
    from lsst.ts.weathernbeats.model import WeatherForecastModel

    # Keep the network tiny for a fast smoke test.
    train_mod.NBEATS_MAX_STEPS = 20
    train_mod.MIN_SLOT_SAMPLES = 3

    grid = train_mod.load_training_grid(None)  # MockClient telemetry
    out = tmp_path / "bundle"
    bundle = train_mod.retrain(None, out, n_slots=48, stride=4)
    assert (out / "metadata.json").exists()
    assert bundle.slots

    reloaded = ModelBundle.load(out)
    model = WeatherForecastModel(reloaded)
    curve = model.predict(grid)
    assert curve["T_forecast"].notna().all()
    assert np.isfinite(curve["T_forecast"].to_numpy()).all()
