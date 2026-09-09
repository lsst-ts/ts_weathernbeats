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

"""Tests for the equinox-label <-> phi <-> date-D wall-clock conversions."""

import numpy as np
import pandas as pd
import pytest

from lsst.ts.weathernbeats.solar_slots import (
    equinox_label_to_phi,
    phi_to_clock_on_date,
    phi_to_equinox_label,
    snap_to_slot,
    solar_slots,
)


def test_equinox_landmark_labels():
    """Solar landmarks map to their equinox-day clock labels."""
    assert phi_to_equinox_label(0.25) == "1200"  # solar midday
    assert phi_to_equinox_label(0.50) == "1800"  # sunset
    # 18:30 just past sunset on the night branch.
    assert phi_to_equinox_label(0.50 + 0.5 / 24.0) == "1830"


def test_label_phi_round_trip():
    """label -> phi -> label is the identity for the canonical slots."""
    for label in ("1200", "1430", "1800", "1830"):
        phi = equinox_label_to_phi(label)
        assert phi_to_equinox_label(phi) == label


def test_snap_to_nearest_slot():
    slots = solar_slots(48)
    idx, phi = snap_to_slot(0.26, slots)
    assert phi == pytest.approx(0.25, abs=1e-9)
    assert slots[idx] == pytest.approx(0.25, abs=1e-9)


def _daytime_hours_per_phi(date):
    """Daytime wall-clock hours spanned per unit of solar coordinate phi.

    On the idealized equinox geometry (12 h day) this equals 24 -- solar time
    advances at clock rate, the basis of the equinox-day labels.
    """
    t0 = phi_to_clock_on_date(0.10, date)
    t1 = phi_to_clock_on_date(0.40, date)
    return (t1 - t0).total_seconds() / 3600.0 / (0.40 - 0.10)


def test_equinox_solar_rate_matches_clock():
    """On an equinox solar time advances at clock rate (~24 h per unit phi).

    This is the property that makes solar time coincide with clock time on the
    equinox, so the equinox-day labels are meaningful.
    """
    equinox = pd.Timestamp("2026-03-20")  # March equinox, ~12 h day
    assert _daytime_hours_per_phi(equinox) == pytest.approx(24.0, abs=0.5)


def test_solstice_rate_shifts():
    """On a solstice the daytime solar rate departs from the equinox rate."""
    equinox = pd.Timestamp("2026-03-20")
    winter = pd.Timestamp("2026-06-21")  # austral winter solstice, short day
    summer = pd.Timestamp("2026-12-21")  # austral summer solstice, long day
    eq = _daytime_hours_per_phi(equinox)
    # Short winter day -> fewer clock hours per unit phi; long summer day -> more.
    assert _daytime_hours_per_phi(winter) < eq - 1.0
    assert _daytime_hours_per_phi(summer) > eq + 1.0


def test_daytime_phi_monotonic_on_date():
    """Wall-clock time increases with phi through the day on a fixed date."""
    date = pd.Timestamp("2026-06-21")
    times = [phi_to_clock_on_date(p, date).value for p in np.linspace(0.05, 0.45, 5)]
    assert all(np.diff(times) > 0)
