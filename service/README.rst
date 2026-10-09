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

The image includes the pinned ``model-v0.2.0`` release bundle at ``/opt/models/nbeatsx_ridge_v0.2.0``.
Run it with the bundled model::

    docker run -p 8080:8080 \
        -e WEATHERNBEATS_SIMULATION=true weathernbeats

To use a different bundle, mount it read-only and set ``WEATHERNBEATS_BUNDLE`` to its path in the container.
The Dockerfile's ``MODEL_VERSION`` and ``MODEL_SHA256`` build arguments select the release asset and verify its SHA-256 digest.
Update their defaults for a new model, or override both with ``docker build --build-arg``.

GitHub Actions builds the image and pushes it to ``ghcr.io/lsst-ts/ts_weathernbeats`` for releases and for pull requests from ``tickets/`` branches.

Publishing the model bundle
===========================

Before publishing a ``model-v*`` release, merge the model-release exclusion in ``.github/workflows/service.yaml`` into ``develop`` through a pull request.
Otherwise, publishing the model release will also start a service image build.

From the repository root, confirm that the bundle's ``metadata.json`` has ``model_version`` set to ``0.2.0``.
Then archive the whole directory and check its SHA-256 digest::

    tar -czf /tmp/nbeatsx_ridge_v0.2.0.tar.gz \
        -C service/bundles nbeatsx_ridge_v0.2.0
    shasum -a 256 /tmp/nbeatsx_ridge_v0.2.0.tar.gz

Copy the resulting digest into the ``MODEL_SHA256`` argument default in ``service/Dockerfile``.
Then upload that exact archive, without regenerating it, to a new GitHub release::

    gh release create model-v0.2.0 /tmp/nbeatsx_ridge_v0.2.0.tar.gz \
        --repo lsst-ts/ts_weathernbeats --target develop \
        --title "Model v0.2.0" \
        --notes "NBEATSx+Ridge model bundle v0.2.0" --latest=false

GitHub CLI creates the tag at ``develop`` if it does not already exist, adds the archive as a release asset, and publishes the release.
For later models, use a new ``model-v*`` tag and archive name, then update the ``MODEL_VERSION`` and ``MODEL_SHA256`` argument defaults in the Dockerfile.
Publish the asset before running a service image build that refers to it.
