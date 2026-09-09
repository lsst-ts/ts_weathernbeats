################
ts_weathernbeats
################

Inference-only two-stage **NBEATSx + per-solar-slot Ridge** twilight
temperature forecaster for the Vera C. Rubin Observatory.

Given recent local weather telemetry, the package preprocesses the data onto a
48-step solar-time grid, runs a pre-trained NBEATSx network to produce a
continuous absolute-temperature forecast, applies the pre-trained per-solar-slot
Ridge correctors to that whole curve, and returns the calibrated continuous
forecast.  The operational values -- twilight, the 3 h dome-opening lead and the
9 h morning HVAC lead -- are read off this curve.

The package does **not** train or refit at inference time; it only loads
artifacts and predicts.  Re-training the artifacts (direct-temperature NBEATSx
plus the per-solar-slot Ridge layer) lives in a separate ``train`` script.

Usage
=====

Run a single forecast from a trained bundle::

    run_weathernbeats --bundle /path/to/bundle [--simulation-mode 1]

Regenerate the artifacts from telemetry::

    train_weathernbeats --csv telemetry.csv --output /path/to/bundle

The forecaster method is validated in the SPIE paper (``docs/spie_nbeatsx.tex``).

Serving predictions over HTTP
==============================

``ts_weatherforecast`` (or anything else, e.g. LOVE) can fetch predicted
temperatures via a single API call instead of importing this package::

    pip install -e ".[service]"
    export WEATHERNBEATS_BUNDLE=/path/to/bundle
    export WEATHERNBEATS_SIMULATION=1   # 0 (default) queries the live EFD
    serve_weathernbeats                 # runs on 0.0.0.0:8080 (set $PORT to change)

Then::

    curl "http://localhost:8080/forecast"               # full horizon curve
    curl "http://localhost:8080/forecast?time=2026-09-08T04:00:00"  # one value

``GET /forecast`` always predicts from the latest EFD telemetry. Omit
``time`` for the full curve (every horizon step: ``time``, ``temperature``,
``std``); pass an ISO8601 UTC ``time`` for a single interpolated value at any
moment in the horizon -- the same read-off used internally for the
twilight/dome/HVAC setpoints, generalized to any timestamp.
