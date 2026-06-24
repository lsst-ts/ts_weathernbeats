.. _version_history:Version_History:

===============
Version History
===============

.. towncrier release notes start

v0.1.0 (2026-06-23)
===================

New Features
------------

- First implementation of ``ts_weathernbeats``: an inference-only, two-stage
  NBEATSx + per-solar-slot Ridge twilight temperature forecaster.

  - ``FeatureBuilder`` acquires temperature telemetry from the EFD (reusing the
    ``ts_weatherforecast`` query pattern, with a ``MockClient`` fallback) and
    builds the 48-step solar-grid feature frame.
  - ``WeatherForecastModel`` loads pre-trained artifacts and runs the direct-
    temperature inference path (NBEATSx continuous forecast -> per-solar-slot
    Ridge correction -> calibrated continuous curve), exposing the operational
    twilight, 3 h dome-opening and 9 h morning read-offs.
  - ``solar_slots`` provides the equinox-label <-> solar-time <-> date wall-clock
    conversions and nearest-slot selection used by both training and inference.
  - A standalone ``train`` script and ``train_weathernbeats`` CLI regenerate the
    bundled artifacts, including the new per-solar-slot Ridge training layer.
  - A ``run_weathernbeats`` CLI runs a single forecast.
