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

"""Per-solar-slot bookkeeping: equinox-label <-> solar-time phi <-> date-D
wall-clock conversions.

A *slot* is a fixed solar-time position ``phi`` in ``[0, 1)`` (0 = sunrise,
0.25 = solar midday, 0.5 = sunset, 0.75 = solar midnight), labelled for humans
by its **equinox-day local clock time**.  On an equinox the 12-hour day makes
solar time coincide with clock time (sunrise = 06:00, sunset = 18:00), so::

    phi = 0.25 -> 12:00   (solar midday)
    phi = 0.50 -> 18:00   (sunset)
    phi ~ 0.52 -> 18:30   (just past sunset, night branch)

On any other date the same ``phi`` maps to a *different* wall-clock time via
that date's actual sunrise/sunset.  Slot selection at inference is always done
**by phi**, never by raw wall-clock time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .solar_grid import STEPS_PER_DAY, get_sun_altitude

__all__ = [
    "CHILE_TZ",
    "EQUINOX_SUNRISE_H",
    "EQUINOX_SUNSET_H",
    "solar_slots",
    "phi_to_equinox_label",
    "equinox_label_to_phi",
    "snap_to_slot",
    "sun_events_on_date",
    "phi_to_clock_on_date",
]

CHILE_TZ = "America/Santiago"
# Idealized equinox geometry: a 12-hour day with sunrise at 06:00 and sunset at
# 18:00 local time.  This anchors the human-readable slot labels.
EQUINOX_SUNRISE_H = 6.0
EQUINOX_SUNSET_H = 18.0


def solar_slots(n: int = STEPS_PER_DAY) -> np.ndarray:
    """Return the canonical solar-time slot positions ``phi``.

    Parameters
    ----------
    n : `int`, optional
        Number of slots per solar day (default 48, the grid cadence).

    Returns
    -------
    `numpy.ndarray`
        ``phi`` values ``[0, 1/n, ..., (n-1)/n]``.
    """
    return np.arange(n, dtype=float) / n


def _phi_to_equinox_hours(phi: float) -> float:
    """Equinox-day local clock time (decimal hours) for a solar coordinate.

    Daytime branch (``phi`` in ``[0, 0.5]``)::

        t = 06:00 + 24 h * phi          (12:00 at phi=0.25, 18:00 at phi=0.5)

    Night branch (``phi`` in ``[0.5, 1)``)::

        t = 18:00 + 24 h * (phi - 0.5)  (e.g. 18:30 at phi~0.52)
    """
    if phi < 0.5:
        return EQUINOX_SUNRISE_H + 24.0 * phi
    return EQUINOX_SUNSET_H + 24.0 * (phi - 0.5)


def phi_to_equinox_label(phi: float) -> str:
    """Return the ``"HHMM"`` equinox-day clock label for solar position ``phi``.

    Hours wrap modulo 24 (e.g. solar midnight ``phi=0.75`` -> ``"0000"``).
    """
    hours = _phi_to_equinox_hours(float(phi)) % 24.0
    minute_of_day = int(round(hours * 60.0)) % (24 * 60)
    return f"{minute_of_day // 60:02d}{minute_of_day % 60:02d}"


def equinox_label_to_phi(label: str) -> float:
    """Invert :func:`phi_to_equinox_label`.

    ``label`` is a 24-hour ``"HHMM"`` string.  Local clock times at or after
    18:00 are interpreted as the night branch; earlier times wrapped past
    midnight (00:00-06:00) are treated as late-night and mapped onto the night
    branch as well.
    """
    label = label.strip()
    hours = int(label[:-2]) + int(label[-2:]) / 60.0
    if EQUINOX_SUNRISE_H <= hours <= EQUINOX_SUNSET_H:
        return (hours - EQUINOX_SUNRISE_H) / 24.0
    # Night branch: wrap pre-dawn hours (< 06:00) to the 24-30 h range.
    if hours < EQUINOX_SUNRISE_H:
        hours += 24.0
    return 0.5 + (hours - EQUINOX_SUNSET_H) / 24.0


def snap_to_slot(phi: float, slots: np.ndarray | None = None) -> tuple[int, float]:
    """Snap a solar coordinate to the nearest defined slot.

    Parameters
    ----------
    phi : `float`
        Solar-time coordinate in ``[0, 1)``.
    slots : `numpy.ndarray`, optional
        Slot positions; defaults to :func:`solar_slots`.

    Returns
    -------
    index : `int`
        Index of the nearest slot.
    slot_phi : `float`
        The nearest slot's ``phi`` value.
    """
    if slots is None:
        slots = solar_slots()
    idx = int(np.argmin(np.abs(slots - float(phi))))
    return idx, float(slots[idx])


def sun_events_on_date(
    date: pd.Timestamp | str,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return ``(sunrise, sunset, next_sunrise)`` for a local calendar date.

    Sun altitude is sampled on a 1-minute grid (vectorized astropy
    `~astropy.time.Time`) across the local day and the following morning; the
    geometric horizon crossings (altitude = 0) give the events.  All returned
    timestamps are tz-aware Chile-local.
    """
    day = pd.Timestamp(date).tz_localize(None).normalize()
    start_local = day.tz_localize(CHILE_TZ)
    # Cover the full local day plus the next morning to capture next sunrise.
    local_grid = pd.date_range(start_local, periods=48 * 60, freq="min")
    utc_naive = pd.DatetimeIndex(local_grid.tz_convert("UTC").tz_localize(None))
    alt = get_sun_altitude(utc_naive)

    rising = np.where((alt[:-1] < 0) & (alt[1:] >= 0))[0]
    setting = np.where((alt[:-1] >= 0) & (alt[1:] < 0))[0]
    if len(rising) < 2 or len(setting) < 1:
        raise RuntimeError(f"Could not bracket sun events on {day.date()}.")

    sunrise = local_grid[rising[0]]
    # First sunset after sunrise, and the following sunrise.
    sunset = local_grid[setting[setting > rising[0]][0]]
    next_sunrise = local_grid[rising[rising > setting[setting > rising[0]][0]][0]]
    return sunrise, sunset, next_sunrise


def phi_to_clock_on_date(phi: float, date: pd.Timestamp | str) -> pd.Timestamp:
    """Map a solar coordinate ``phi`` to a wall-clock time on a given date.

    Daytime (``phi`` in ``[0, 0.5]``)::

        t = sunrise + 2 * phi * (sunset - sunrise)

    Night (``phi`` in ``[0.5, 1)``)::

        t = sunset + (2 * phi - 1) * (next_sunrise - sunset)

    Returns a tz-aware Chile-local timestamp.
    """
    sunrise, sunset, next_sunrise = sun_events_on_date(date)
    phi = float(phi)
    if phi < 0.5:
        return sunrise + 2.0 * phi * (sunset - sunrise)
    return sunset + (2.0 * phi - 1.0) * (next_sunrise - sunset)
