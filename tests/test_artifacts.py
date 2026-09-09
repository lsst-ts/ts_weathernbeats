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

"""Tests for the per-slot Ridge artifact I/O (NBEATSx save/load is exercised
in the retrain integration test, which requires PyTorch)."""

import joblib

from lsst.ts.weathernbeats.artifacts import SlotModel


def test_slot_model_round_trip(tmp_path, fake_bundle):
    """A slot dict round-trips through joblib with scaler + ridge intact."""
    path = tmp_path / "ridge_slots.joblib"
    joblib.dump(fake_bundle.slots, path)
    loaded = joblib.load(path)

    assert set(loaded) == set(fake_bundle.slots)
    a_phi = next(iter(loaded))
    slot = loaded[a_phi]
    assert isinstance(slot, SlotModel)
    assert slot.equinox_label
    assert slot.feature_list
    assert hasattr(slot.scaler, "transform")
    assert hasattr(slot.ridge, "predict")


def test_slot_self_describing(fake_bundle):
    """Each slot stores phi, scaler, feature schema and equinox label."""
    for phi, slot in fake_bundle.slots.items():
        assert slot.phi == phi
        assert slot.feature_list == fake_bundle.metadata["ridge_features"]
        assert isinstance(slot.equinox_label, str)


def test_bundle_slot_phis_sorted(fake_bundle):
    phis = fake_bundle.slot_phis
    assert phis == sorted(phis)
