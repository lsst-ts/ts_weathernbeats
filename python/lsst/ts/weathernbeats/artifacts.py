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

"""Bundled artifact I/O for the two-stage NBEATSx + per-slot Ridge forecaster.

A bundle is a single directory holding everything inference needs::

    bundle/
        metadata.json           model version, training window, lib versions
        nbeatsx/                NeuralForecast.save() directory (Stage 1)
        ridge_slots.joblib      {phi: SlotModel} keyed by solar slot (Stage 2)

Each :class:`SlotModel` carries its own scaler, fitted Ridge, ``phi``,
feature schema and equinox-time label so inference is fully self-describing
and never parses times out of filenames.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import joblib

__all__ = ["SlotModel", "ModelBundle", "NBEATSX_SUBDIR", "RIDGE_FILE", "METADATA_FILE"]

NBEATSX_SUBDIR = "nbeatsx"
RIDGE_FILE = "ridge_slots.joblib"
METADATA_FILE = "metadata.json"


@dataclasses.dataclass
class SlotModel:
    """A per-solar-slot Ridge corrector with its scaler and schema.

    Attributes
    ----------
    phi : `float`
        Solar-time coordinate of the slot in ``[0, 1)``.
    equinox_label : `str`
        Human-readable equinox-day clock label (``"HHMM"``).
    feature_list : `list` [`str`]
        Ordered Ridge feature columns (excluding the appended NBEATSx
        prediction, which is always the final input).
    scaler : `sklearn.preprocessing.StandardScaler`
        Fitted feature scaler.
    ridge : `sklearn.linear_model.Ridge`
        Fitted corrector predicting absolute temperature.
    residual_std : `float`
        Standard deviation of the corrector's training residuals
        (measured minus corrected), used as the forecast uncertainty band.
    """

    phi: float
    equinox_label: str
    feature_list: list[str]
    scaler: Any
    ridge: Any
    residual_std: float = 0.5


@dataclasses.dataclass
class ModelBundle:
    """A loaded two-stage model: NBEATSx + per-slot Ridge correctors."""

    nbeatsx: Any
    slots: dict[float, SlotModel]
    metadata: dict[str, Any]

    @property
    def slot_phis(self) -> list[float]:
        """Sorted list of the defined slot ``phi`` positions."""
        return sorted(self.slots)

    def save(self, path: str | Path) -> None:
        """Persist the bundle to ``path`` (created if absent)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.nbeatsx.save(
            str(path / NBEATSX_SUBDIR), overwrite=True, save_dataset=False
        )
        joblib.dump(self.slots, path / RIDGE_FILE)
        (path / METADATA_FILE).write_text(json.dumps(self.metadata, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        """Load a bundle previously written by :meth:`save`."""
        from neuralforecast import NeuralForecast

        path = Path(path)
        nbeatsx = NeuralForecast.load(str(path / NBEATSX_SUBDIR))
        slots: dict[float, SlotModel] = joblib.load(path / RIDGE_FILE)
        metadata = json.loads((path / METADATA_FILE).read_text())
        return cls(nbeatsx=nbeatsx, slots=slots, metadata=metadata)
