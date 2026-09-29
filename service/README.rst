#####################
weathernbeats service
#####################

``weathernbeats`` is a `Safir <https://safir.lsst.io>`__ FastAPI application that publishes the ``ts_weathernbeats`` temperature forecast over HTTP, for deployment with `Phalanx <https://phalanx.lsst.io>`__.
It loads the model bundle once at startup; each request then queries the EFD for the latest temperature telemetry and runs the NBEATSx+Ridge forecast.

API
===

``GET /weathernbeats/forecast``
    The calibrated forecast over the whole horizon, about half a solar day ahead: one point per solar-grid step, each with ``time`` (UTC), ``temperature`` and ``std`` (degrees Celsius).
``GET /weathernbeats/forecast?time=2026-03-21T04:00:00Z``
    The forecast temperature interpolated at ``time``.
    Times without a timezone are taken to be UTC.
    A time outside the horizon returns a 422 error that states the horizon.
``GET /weathernbeats/docs``
    Interactive API documentation.
``GET /``
    Application metadata, used as the Kubernetes health check.
    This route is not exposed outside the cluster.

Configuration
=============

The service is configured with environment variables.

``WEATHERNBEATS_BUNDLE`` (required)
    Directory holding a trained model bundle, such as ``/sdf/group/rubin/web_data/guider-diagnostics/ts_weathernbeats/models/nbeatsx_ridge_v0.2.0``.
``WEATHERNBEATS_SIMULATION``
    If ``true``, forecast from synthetic telemetry instead of the EFD (default ``false``).
``LSST_SITE``
    Selects the EFD instance (``summit``, ``base``, ``usdf`` or ``tucson``), as in the rest of ``ts_weathernbeats``.
    The EFD client fetches its credentials from Segwarides, even in simulation mode.
``WEATHERNBEATS_PATH_PREFIX``
    URL prefix of the API (default ``/weathernbeats``).
``WEATHERNBEATS_LOG_LEVEL``, ``WEATHERNBEATS_LOG_PROFILE``
    Safir logging settings (defaults ``INFO`` and ``development``; ``production`` logs JSON).
``WEATHERNBEATS_SLACK_WEBHOOK``
    Optional Slack webhook for alerts about uncaught exceptions.

Development
===========

The service is its own uv project, separate from the conda and EUPS packaging of the library at the top of the repository.
It installs the library from the repository root as an editable path dependency.

Set up the environment and run the lint, typing and test sessions::

    cd service
    make init
    uv run nox

Run a local server against a bundle, with synthetic telemetry::

    export WEATHERNBEATS_BUNDLE=/path/to/nbeatsx_ridge_v0.2.0
    export WEATHERNBEATS_SIMULATION=true
    make run
    curl "http://localhost:8000/weathernbeats/forecast"

Update the pinned dependencies with ``make update``.

Docker image
============

The build context is the repository root, because the image installs the library from there::

    docker build -f service/Dockerfile -t weathernbeats .

The image does not contain a model bundle.
Mount one and point ``WEATHERNBEATS_BUNDLE`` at it::

    docker run -p 8080:8080 -v /path/to/models:/models:ro \
        -e WEATHERNBEATS_BUNDLE=/models/nbeatsx_ridge_v0.2.0 \
        -e WEATHERNBEATS_SIMULATION=true weathernbeats

GitHub Actions builds the image and pushes it to ``ghcr.io/lsst-ts/ts_weathernbeats`` for releases and for pull requests from ``tickets/`` branches.
