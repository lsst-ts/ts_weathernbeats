.. py:currentmodule:: lsst.ts.weathernbeats

.. _lsst.ts.weathernbeats:

#####################
lsst.ts.weathernbeats
#####################

Inference-only two-stage NBEATSx + per-solar-slot Ridge twilight temperature
forecaster.  Recent weather telemetry is resampled onto a 48-step solar-time
grid, a pre-trained NBEATSx network produces a continuous absolute-temperature
forecast, and pre-trained per-solar-slot Ridge correctors calibrate the whole
curve; the operational twilight, 3 h dome-opening and 9 h morning values are
read off the corrected curve.

.. toctree::
   :maxdepth: 1

   integration
   training
   version_history

.. .. _lsst.ts.weathernbeats-using:

.. Using lsst.ts.weathernbeats
.. ===========================

.. toctree linking to topics related to using the module's APIs.

.. .. toctree::
..    :maxdepth: 1

.. _lsst.ts.weathernbeats-contributing:

Contributing
============

``lsst.ts.weathernbeats`` is developed at https://github.com/lsst-ts/ts_weathernbeats.
You can find Jira issues for this module under the `ts_weathernbeats <https://jira.lsstcorp.org/issues/?jql=project%20%3D%20DM%20AND%20component%20%3D%20ts_weathernbeats>`_ component.

.. If there are topics related to developing this module (rather than using it), link to this from a toctree placed here.

.. .. toctree::
..    :maxdepth: 1

.. .. _lsst.ts.weathernbeats-scripts:

.. Script reference
.. ================

.. .. TODO: Add an item to this toctree for each script reference topic in the scripts subdirectory.

.. .. toctree::
..    :maxdepth: 1

.. .. _lsst.ts.weathernbeats-pyapi:

Python API reference
====================

.. automodapi:: lsst.ts.weathernbeats
   :no-main-docstr:
   :no-inheritance-diagram:
