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

"""Solar-time grid utilities for causal twilight forecasting.

Ported from ``rubin-twilight-forecast/twilight/features.py`` and
``RubinsOraclePaper/forecast/analysis/solar_grid.py``.  Provides sun-altitude
evaluation, solar-event detection, normalized solar-time coordinates and
resampling onto a regular solar-time grid.

The solar-time coordinate ``SolarTime`` lives in ``[0, 1)`` with
``0`` = sunrise, ``0.25`` = solar midday, ``0.5`` = sunset and ``0.75`` =
solar midnight; ``DayCount`` increments at every sunrise and
``solarDayHour = DayCount + SolarTime`` is monotonically increasing.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time

__all__ = [
    "RUBIN_SITE",
    "CHILE_TZ",
    "SUN_ALT_MIDPOINT",
    "STEPS_PER_DAY",
    "SOLAR_GRID_STEP",
    "get_sun_altitude",
    "detect_events",
    "solar_time",
    "add_solar_time",
    "resample_to_solar_grid",
    "find_twilight_targets",
]

# Observatory site (Cerro Pachón).
RUBIN_SITE = EarthLocation(
    lat=-30.2446 * u.deg,
    lon=-70.7494 * u.deg,
    height=2663.0 * u.m,
)
CHILE_TZ = "America/Santiago"
# Sun altitude (deg) defining the operational twilight event used by the
# shipped artifacts.  The SPIE evaluation targets astronomical twilight; the
# reference training pipeline uses -20 deg (dome-opening twilight).
SUN_ALT_MIDPOINT = -20.0

# 48 grid steps per solar day -> 30-minute equivalent cadence.
STEPS_PER_DAY = 48
SOLAR_GRID_STEP = 1.0 / STEPS_PER_DAY


def get_sun_altitude(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Return the sun altitude in degrees at Rubin for UTC ``timestamps``.

    Parameters
    ----------
    timestamps : `pandas.DatetimeIndex`
        Timezone-naive (UTC) timestamps.

    Returns
    -------
    `numpy.ndarray`
        Sun altitude in degrees, one value per timestamp.  Vectorized over the
        whole index via astropy `~astropy.time.Time`.
    """
    times = Time(timestamps.values)
    altaz = AltAz(obstime=times, location=RUBIN_SITE)
    return get_sun(times).transform_to(altaz).alt.deg


def detect_events(
    df: pd.DataFrame, alt_col: str = "alt_sun", midpoint: float = SUN_ALT_MIDPOINT
) -> pd.DataFrame:
    """Set solar-event flags from zero-crossings of the sun altitude.

    Adds ``event_sunrise`` (altitude crosses 0 rising), ``event_sunset``
    (crosses 0 setting) and ``twilight_event_sunset`` (crosses ``midpoint``
    setting).
    """
    df = df.copy()
    alt = df[alt_col]
    alt_prev = alt.shift(1)

    df["event_sunrise"] = (alt_prev < 0) & (alt >= 0)
    df["event_sunset"] = (alt_prev >= 0) & (alt < 0)

    adjusted = alt - midpoint
    adjusted_prev = adjusted.shift(1)
    df["twilight_event_sunset"] = (adjusted_prev >= 0) & (adjusted < 0)

    return df


def _sun_event_anchors(
    start: pd.Timestamp, end: pd.Timestamp, pad_hours: float = 36.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(event_seconds, event_solar)`` for sun events bracketing a span.

    Sun-altitude zero-crossings (sunrise rising, sunset setting) are found on a
    1-minute astropy grid padded by ``pad_hours`` either side of ``[start,
    end]`` -- so events *beyond* the data window (e.g. the next sunrise after a
    night that the data does not yet reach) are still available.  Each event is
    assigned a monotonically increasing solar coordinate: consecutive events are
    +0.5 apart (sunrise <-> sunset), with sunrise on integer values and sunset
    on the half-integers.  Interpolating a timestamp against these anchors gives
    SolarTime with sunrise=0.0, sunset=0.5, solar-midnight=0.75 exactly.
    """
    grid = pd.date_range(
        pd.Timestamp(start) - pd.Timedelta(hours=pad_hours),
        pd.Timestamp(end) + pd.Timedelta(hours=pad_hours),
        freq="min",
    )
    alt = get_sun_altitude(grid)
    prev = np.concatenate([[alt[0]], alt[:-1]])
    rising = (prev < 0) & (alt >= 0)
    setting = (prev >= 0) & (alt < 0)
    ev_mask = rising | setting
    ev_idx = np.where(ev_mask)[0]
    ev_sec = grid[ev_idx].astype("int64").to_numpy() / 1e9
    # Solar coordinate: sunrise -> *.0, sunset -> *.5.  Anchor the first event to
    # the nearest half-day boundary by its kind, then step +0.5 per event.
    ev_solar = 0.5 * np.arange(len(ev_idx), dtype=float)
    if rising[ev_idx[0]]:
        ev_solar += 0.0  # first event is a sunrise -> integer
    else:
        ev_solar += 0.5  # first event is a sunset -> half-integer
    return ev_sec, ev_solar


def solar_time(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Continuous solar time (``solarDayHour``) for UTC ``timestamps``.

    The standard, edge-robust solar-time map: ``floor`` is the solar-day count
    and the fractional part is ``SolarTime`` in ``[0, 1)`` with sunrise=0.0,
    solar-midday=0.25, sunset=0.5, solar-midnight=0.75.  Built by interpolating
    against real astropy sun events (:func:`_sun_event_anchors`), so it is
    correct right up to the last timestamp -- including a night that runs past
    the end of the observed data.
    """
    ts = pd.DatetimeIndex(timestamps)
    ev_sec, ev_solar = _sun_event_anchors(ts.min(), ts.max())
    x = ts.astype("int64").to_numpy() / 1e9
    return np.interp(x, ev_sec, ev_solar)


def add_solar_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``SolarTime``, ``DayCount`` and ``solarDayHour`` columns.

    Uses :func:`solar_time` (astropy sun-event anchors), so SolarTime is exact
    at the landmarks (sunrise=0.0, sunset=0.5, solar-midnight=0.75) and is
    defined for every row -- including the trailing partial night that the old
    in-window-event method dropped for lack of a "next sunrise" in the data.
    """
    df = df.copy()
    ds = pd.DatetimeIndex(pd.to_datetime(df["ds"]))

    sdh = solar_time(ds)
    # Anchor the count so the first *sunrise* in range is DayCount 0: drop the
    # leading partial day (rows before the first sunrise, i.e. before the first
    # integer solar coordinate) so solarDayHour is non-negative and starts at a
    # sunrise, exactly as the resampler's monotonic-grid assumption expects.
    first_sunrise = np.ceil(sdh[0])
    sdh = sdh - first_sunrise
    df["solarDayHour"] = sdh
    df["DayCount"] = np.floor(sdh).astype(int)
    df["SolarTime"] = sdh - df["DayCount"]

    df = df[df["solarDayHour"] >= 0].copy()
    return df


def resample_to_solar_grid(
    df: pd.DataFrame, step: float = SOLAR_GRID_STEP, fillna: bool = True
) -> pd.DataFrame:
    """Interpolate all numeric features onto a regular ``solarDayHour`` grid.

    ``solarDayHour`` is monotonically increasing by construction; we sort and
    drop duplicate x-values as a guard (`numpy.interp` requires strictly
    increasing x).  A synthetic regular solar clock is regenerated as ``ds``
    (each solar day = 24 h, sunrise at 06:00) while the real UTC wall-clock
    time is preserved as ``ds_real``.  ``step = 1/48`` gives 48 points per
    solar day.
    """
    df = df.sort_values("solarDayHour")
    df = df[~df["solarDayHour"].duplicated(keep="first")]

    x = df["solarDayHour"].to_numpy(dtype=float)
    grid = np.arange(0.0, float(x.max()), step)

    out = pd.DataFrame({"solarDayHour": grid})
    out["DayCount"] = np.floor(grid).astype(int)
    out["SolarTime"] = grid - out["DayCount"]

    drop = {
        "solarDayHour", "DayCount", "SolarTime", "ds", "ds_real", "ds_local",
        "event_sunrise", "event_sunset", "twilight_event_sunset",
    }
    num_cols = [
        c for c in df.columns
        if c not in drop and pd.api.types.is_numeric_dtype(df[c])
    ]
    for c in num_cols:
        out[c] = np.interp(grid, x, df[c].to_numpy(dtype=float))

    # Synthetic regular solar clock: 06:00 on the date of the first real
    # sunrise, advanced by exactly one grid step per row (integer index avoids
    # float drift that would make NeuralForecast reject futr_df).
    first_sunrise = pd.Timestamp(df["ds"].iloc[0])
    base = first_sunrise.normalize() + pd.Timedelta(hours=6)
    step_delta = pd.Timedelta(seconds=round(step * 24.0 * 3600.0))
    out["ds"] = base + np.arange(len(grid)) * step_delta

    # Preserve the actual wall-clock time (int64-ns round-trip).
    ds_ns = pd.to_datetime(df["ds"]).astype("int64").to_numpy()
    out["ds_real"] = pd.to_datetime(np.interp(grid, x, ds_ns).astype("int64"))

    if fillna:
        out[num_cols] = out[num_cols].ffill().bfill()

    # Regenerate event flags from the interpolated alt_sun (booleans can't be
    # interpolated, but downstream feature engineering needs them).
    if "alt_sun" in out.columns:
        out = detect_events(out)

    return out


def find_twilight_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Find grid rows where ``alt_sun`` crosses the twilight midpoint setting.

    Returns one row per twilight event with the grid index, solar coordinate,
    real timestamp, day count and the measured temperature at the event.
    """
    mask = df["twilight_event_sunset"].values.astype(bool)
    idxs = np.where(mask)[0]

    events = []
    for i in idxs:
        events.append({
            "grid_idx": int(i),
            "solarDayHour": df["solarDayHour"].iloc[i],
            "ds_real": df["ds_real"].iloc[i],
            "DayCount": df["DayCount"].iloc[i],
            "y_actual": df["y"].iloc[i],
        })
    return pd.DataFrame(events)
