.. py:currentmodule:: lsst.ts.weathernbeats

.. _lsst.ts.weathernbeats-integration:

########################################
Integrating with the WeatherForecast CSC
########################################

This describes how ``ts_weatherforecast`` should consume this package's
forecasts, replacing its current in-process Prophet fit with an HTTP call to
the forecast service (``serve_weathernbeats``, see the package README).

Why this matters beyond WeatherForecast itself: ``tel_hourlyTrend.temperature``
is what LOVE (the operator UI) reads to display the forecast on the summit
dashboard. Operators see whatever this integration produces directly --
including the horizon/cadence gap called out below, if it isn't resolved.

What ``ts_weatherforecast`` does today
=======================================

Two independent pieces publish telemetry from the CSC
(``python/lsst/ts/weatherforecast/csc.py``):

* **Meteoblue** (``write_data`` / ``get_response``) polls an external weather
  API twice a day and publishes ``tel_hourlyTrend`` / ``tel_dailyTrend`` --
  wind, humidity, pressure, cloud cover, and ~30 other fields. This is
  unrelated to temperature forecasting and is **not** replaced by this
  package.
* **Prophet** (``BobDobbs`` in ``model.py``, driven by ``prediction_loop`` in
  ``csc.py``) queries the same EFD temperature series this package uses
  (``lsst.sal.ESS.temperature``, ``salIndex=301``, 7-day window), fits a
  Prophet model **in-process every 15 minutes**, and publishes
  ``prediction["yhat"]`` -- 288 points at 5-minute cadence (24 h ahead) -- as
  ``tel_hourlyTrend.temperature``.

The Prophet piece is what this package's forecast service replaces.

What changes
============

Instead of fitting Prophet in-process, ``make_prediction`` should call this
service's ``GET /forecast`` endpoint and use its ``curve`` in place of
``prediction["yhat"]``::

    async def make_prediction(self) -> pd.Series:
        async with aiohttp.ClientSession(WEATHERNBEATS_URL) as session:
            async with session.get("/forecast") as resp:
                body = await resp.json()
        curve = body["curve"]  # [{"time": iso8601, "temperature": float, "std": float}, ...]
        return pd.Series([row["temperature"] for row in curve])

``aiohttp`` is already a dependency of ``ts_weatherforecast`` (used for the
Meteoblue call), so no new client library is needed.

Two real differences the CSC side must decide how to handle
=============================================================

This is **not** a drop-in replacement of the array shape:

* **Horizon and cadence differ.** Prophet publishes 288 points on a fixed
  5-minute grid covering 24 h. This service's curve covers ~12.5 h on a
  *solar-time* grid (uneven wall-clock spacing -- roughly 30 min steps that
  compress during the day and stretch at night; see
  :py:meth:`WeatherForecastModel.predict`). If ``tel_hourlyTrend.temperature``
  must stay a fixed-length, fixed-cadence array, the CSC needs to resample
  this curve onto its own grid (e.g. via ``GET /forecast?time=...`` per
  desired timestamp, or by interpolating the full curve client-side) rather
  than publish it as-is.
* **Temperature only.** This service predicts one quantity. It does not
  provide ``temperatureSpread`` or any other ``tel_hourlyTrend`` field --
  those keep coming from Meteoblue, unchanged.

Running the service
====================

The service needs to be deployed and reachable from wherever the CSC runs;
see "Serving predictions over HTTP" in this package's README for how to
start it, required environment variables (``WEATHERNBEATS_BUNDLE``), and the
S3DF/USDF InfiniBand workaround if co-located on that cluster.
