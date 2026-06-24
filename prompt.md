# Task: build `ts_weathernbeats`, an inference-only NBEATSx+Ridge temperature forecaster

You are working inside the LSST Telescope & Site (TSSW) software ecosystem. This is the **first
NBEATSx+Ridge implementation in lsst-ts** — no working version exists yet.

The landscape:
- **`ts_weatherforecast`** is the existing, **working** forecaster: a Prophet-based CSC
  (operational at Rubin prior to this work). It is the source of the engineering conventions,
  package structure, and EFD data-acquisition pattern you will mirror.
- **`ts_weathernbeats`** is currently only an **initial, non-working repo skeleton**. Your job
  is to implement it: an inference-only library that loads the (already trained) NBEATSx +
  per-slot Ridge artifacts and produces calibrated temperature forecasts. Re-training lives in a
  separate script. The NBEATSx+Ridge *method* is validated in the SPIE paper and the trained
  artifacts exist; this package does **not** train or refit at inference time.

Match the engineering conventions of `ts_weatherforecast` exactly.

Do not start writing code until you have completed the "Read first" and "Plan and confirm"
steps below.

---

## 1. Read first (do not write code yet)

Read each of these and extract the noted information:

- **Model description (the science):**
  `/sdf/home/e/esteves/sitcom-analysis/ts_weathernbeats/docs/spie_nbeatsx.tex`
  → the two-stage architecture, the solar-time grid, the feature definitions, and which
  lead times matter operationally (3 h = dome-opening / M1M3 setpoint, 9 h = morning HVAC).

- **Reference implementation (for exact feature/preprocessing details):**
  `/sdf/home/e/esteves/sitcom-analysis/RubinsOraclePaper`
  → use it for concrete details: feature names, solar-grid construction, artifact I/O, the
  NBEATSx forward pass. **Caveat:** the model *formulation* is fixed by §4 (direct-temperature)
  and the SPIE paper, not by whichever script you happen to open. If the code you are reading
  trains NBEATSx on `delta_T` or has Ridge predict `tw_slope`, that is the **older, superseded**
  variant — see the reference-version warning in §4.

- **Existing CSC data-acquisition pattern (reuse this for the FeatureBuilder):**
  `https://github.com/lsst-ts/ts_weatherforecast/blob/d4b2d367eb534b5c5821c5d75a57017c557383af/python/lsst/ts/weatherforecast/model.py`
  (around L65, the `BobDobbs.create_client` / `query` / `setup_fit` methods)
  → this is how the live system gets telemetry from the EFD. The new FeatureBuilder must obtain
  its input data the same way (see §4b). Do not invent a separate CSV-based loader for the
  runtime path.

- **Target package conventions (mimic these):**
  `/sdf/home/e/esteves/sitcom-analysis/ts_weatherforecast`
  → folder layout, namespace-package structure, `pyproject.toml`/`setup.py`/`setup.cfg`,
  ruff/linting config, `doc/` layout (incl. `version_history.rst`), `tests/` layout, `bin/`,
  conda/eups packaging, docstring style, type-hint conventions, CSC structure (if any).

- **Pre-commit / lint config (must pass):**
  `/sdf/home/e/esteves/sitcom-analysis/ts_weatherforecast/.ts_pre_commit_config.yaml`
  `/sdf/home/e/esteves/sitcom-analysis/ts_weathernbeats/setup.cfg`
  → the exact set of hooks and lint rules. Do not guess the hook list; read it and make the
  code satisfy every hook.

- **Target API (replicate its call pattern):**
  `/sdf/home/e/esteves/sitcom-analysis/twilight_forecast_history.ipynb`
  → the public interface the module must expose should make this notebook's usage work with
  minimal changes. Treat the notebook's calls as the API contract.

After reading, briefly state back to me: (a) the existing state of `ts_weathernbeats` (is it
an empty skeleton or partially built?), (b) the trained-model artifact format and location,
and (c) the API shape the notebook expects.

---

## 2. What we are building (one-paragraph summary)

You are implementing — for the first time in lsst-ts — an inference-only library that wraps the
two-stage **NBEATSx + Ridge** forecaster from the SPIE paper. (`ts_weathernbeats` does not work
today; it is an initial skeleton. The existing working forecaster is the Prophet-based
`ts_weatherforecast`.) Given recent local weather telemetry, the library should preprocess the
data onto the solar-time grid, run the **pre-trained** NBEATSx network to produce a
**continuous** temperature forecast, apply the **pre-trained per-solar-slot** Ridge correctors
to that whole curve, and return the calibrated continuous forecast — from which the operational
values (twilight, 3 h dome-opening, 9 h morning) are read off. It must **not** train, fit, or
refit anything at inference time, and it should be well tested when complete.

---

## 3. Scope

**In scope**
- Loading pre-trained model artifacts (NBEATSx + the **per-solar-slot** Ridge models + their
  scalers).
- The full inference path: telemetry in → preprocessing → solar grid → features → NBEATSx
  (continuous forecast) → per-slot Ridge correction → corrected continuous curve out.
- A clean public API matching the notebook, plus a small CLI entry point.
- A **separate** `retrain` / `train` script (or `bin/` command) that regenerates the artifacts,
  including the **new Ridge training layer** that fits one corrector per solar-grid slot (§4c).
  Port the existing training from the reference implementation; do not invest effort in model
  development, tuning, or new architectures.
- Unit tests, docstrings, type hints, `version_history` entry, docs — to TSSW standards.

**Out of scope (do not port these)**
- The Persistence / Linear / Random Forest / MLP baselines.
- Prophet and MeteoBlue comparison/ingest code.
- Figure generation and paper-evaluation scripts.
- Any new model design, hyperparameter sweep, or architecture change.

---

## 4. Reproduce the exact inference path (direct-temperature, continuous output)

NBEATSx emits a **continuous temperature forecast** — a predicted value at every future
solar-grid step over its horizon, not just at twilight. The Ridge layer corrects that whole
curve. The model predicts **absolute temperature directly**: it is not trained on temperature
differences, and Ridge does not output a slope.

1. **Preprocess telemetry.** Resample raw temperature to 15-min cadence using the mid-range
   mean `(T_max + T_min)/2`; replace high-intra-interval-spread points via **causal**
   forward-fill (no future leakage anywhere in the pipeline).
2. **Solar-time grid.** Resample onto the uniform solar-time grid (48 steps per solar day, `φ`
   anchored to the date's sunrise/sunset via the paper's solar-time equation). All lag/trend
   features are computed on this grid.
3. **Stage 1 — NBEATSx (direct-T, continuous).** Run the loaded NBEATSx model forward from the
   48-step lookback window to obtain the predicted **absolute temperature** `T̂_NB(φ)` at every
   step over the horizon. A 24 h difference (`y_diff-24h`) is only an input feature, never the
   target.
4. **Stage 2 — per-slot Ridge correction.** For each forecast target step, identify its
   solar-grid slot `φ` (see §4c), load **that slot's** `StandardScaler` + `Ridge`, build the
   feature vector `x` (the solar-grid features from the paper, plus the NBEATSx prediction
   `T̂_NB` at that step as an input), and apply it.
5. **Final forecast.** `T̂_final(φ) = Ridge_φ(x, T̂_NB(φ))` — Ridge outputs the **absolute
   temperature** directly. There is **no** `T_tw_last + slope` reconstruction. The result is a
   bias-corrected continuous forecast; the operational twilight / 3 h / 9 h values are read off
   this curve.
6. **Operational targets.** Make twilight, the 3 h dome-opening lead, and the 9 h morning lead
   first-class outputs derived from the corrected curve.

Pin the `neuralforecast` / `pytorch` / `pytorch-lightning` / `scikit-learn` versions to whatever
produced the saved artifacts — a version mismatch will silently fail or change
`NeuralForecast.load()` / `joblib.load()` behavior. Confirm the saved models load and reproduce
a known prediction before going further.

> **Reference-version warning.** Some scripts in the codebase use a *delta / slope* formulation
> (NBEATSx trained on `delta_T`; Ridge predicting `tw_slope = tw_temp − T_tw_last`; final
> `= T_tw_last + slope`) and a *single twilight target / per-lead* Ridge. Both are **superseded**.
> The final model uses the **direct-temperature, continuous-output, per-solar-slot** formulation
> above. If you encounter the older variants, do not follow them; confirm which code produced the
> shipped artifacts.

---

## 4c. Per-slot Ridge correctors and the equinox-time convention

The Ridge layer is **one corrector per solar-grid slot** (this replaces the old per-lead-time
correctors). Each slot is a fixed solar-time position `φ`, labeled for humans by its
**equinox-day local clock time** — on an equinox the 12 h day makes solar time coincide with
clock time (sunrise ≈ 06:00, sunset ≈ 18:00). The user's example window spans the
afternoon→evening at 30-min cadence: 12:00 → 18:30.

**New training layer.** For each slot `φ`, gather across all training days the
(NBEATSx prediction at `φ`, issuance-time features, measured temperature at `φ`), fit a
`StandardScaler` + `Ridge`, and persist it keyed by `φ`, with the equinox-time label as metadata.

**Reuse the existing conversion utilities — do not reimplement.** The codebase already has
solar-time / sunrise-sunset conversion helpers, built on **vectorized astropy `Time`** objects
for speed. Locate them first (look in the `ts_weatherforecast` `utils` module — `model.py`
imports `from .utils import …` — and in the reference implementation) and use them for the
`φ` ↔ wall-clock conversion and for sunrise/sunset. The relationships below describe what those
helpers compute, for your verification only:

- Each label ↔ a fixed `φ` from equinox geometry. Daytime branch: `φ = (t_equinox − 06:00)/24 h`
  (12:00 → 0.25, 18:00 → 0.50); night branch past sunset (18:30 → ≈0.52).
- On date `D`, the same `φ` maps to a **different** wall-clock time via `D`'s actual
  sunrise/sunset (daytime: `t = sunrise(D) + 2φ·(sunset(D) − sunrise(D))`; night branch past
  sunset). At inference, compute `φ` for the forecast time(s) on date `D`, then select the slot
  model **by `φ`** — never by raw wall-clock time.
- Make sure the existing util covers both the daytime/night branch split and the equinox
  convention; snap to the nearest defined slot. Unit-test the round trip: on an equinox a slot's
  wall-clock time equals its label; on a solstice it shifts by the expected amount.

**Naming / format (improve on the example).** Per-file `*_12_00pm.joblib` works, but the `pm`
suffix on 24-h times (`18_30pm`) is ambiguous and separate files are easy to desync. Prefer
either unambiguous 24-h solar labels (`nbeatsx_ridge_solar_1200.joblib … _1830.joblib`) or —
cleaner for LSST versioning — a **single bundled artifact** (one directory or joblib) holding
`{slot: {model, scaler, phi, feature_list, equinox_label}}` plus top-level metadata (model
version, training window, sklearn/neuralforecast versions, feature schema). Whichever you pick,
store `φ`, the scaler, and the feature schema **with each slot** so inference is self-describing
and never parses times out of filenames.

---

## 4b. FeatureBuilder data acquisition (reuse the CSC's EFD query)

Re-make the FeatureBuilder so its runtime input comes from the EFD the same way
`ts_weatherforecast/model.py` does (the `BobDobbs.create_client` / `query` / `setup_fit`
methods, ~L65), then layer the solar-grid feature engineering on top of that frame:

- Use `lsst_efd_client.EfdClient`, selecting the client from the `LSST_SITE` environment
  variable, with the same `MockClient` fallback so `simulation_mode` works without a live EFD.
- Query the recent temperature history (the existing code pulls the 1-minute mean of
  `temperatureItem0` from `lsst.sal.ESS.temperature`, salIndex 301, over a ~7-day window with
  `GROUP BY time(1m) FILL(linear)`). Reuse this query as the data source.
- Reshape into the timezone-naive `ds`/`y` frame (as `setup_fit` does), then run the §4
  preprocessing → solar grid → feature pipeline on it.
- **Confirm** the topic/`salIndex`/field actually correspond to the Summit Weather Tower sensor
  the NBEATSx artifacts were trained on; if training used a different sensor, adjust the query
  to match (the model is only valid on its training sensor).
- A ~7-day window comfortably covers the 48-step solar-grid lookback; keep or widen it as the
  feature set requires, but do not switch the runtime path to a CSV loader.

---

## 5. Deliverables

- A TSSW-conformant package tree mirroring `ts_weatherforecast` (namespace layout under
  `python/lsst/ts/...`, `doc/`, `tests/`, `bin/`, packaging files, version history).
- A **FeatureBuilder** that acquires telemetry via the EFD query pattern from §4b and produces
  the solar-grid feature frame the two stages consume.
- A public API that makes `twilight_forecast_history.ipynb` work — propose the exact
  class/function signatures in your plan.
- A CLI entry point for a single forecast run.
- A standalone `retrain` script that regenerates the artifacts from telemetry (direct-T
  target), **including the new per-solar-slot Ridge training layer** that emits one corrector
  per slot in the chosen format (§4c).
- Solar-grid slot selection + the equinox-label ↔ `φ` ↔ date-`D` wall-clock conversion logic
  (§4c), used by both training and inference.
- Tests covering: artifact loading, the FeatureBuilder/EFD path (with `MockClient`),
  preprocessing/solar-grid correctness, **the equinox↔date solar-time conversion** (equinox:
  slot time == label; solstice: shifts as expected), per-slot Ridge selection and the two-stage
  direct-T inference (NBEATSx → per-slot Ridge → absolute-temperature curve), and end-to-end
  inference on a small fixture.
- `doc/version_history.rst` entry and module docs.

---

## 6. Workflow and coding principles

**Coding principles (follow throughout):**
- **YAGNI.** Build only what these requirements call for. No speculative abstractions, config
  knobs, plugin layers, or "future-proofing" the spec doesn't ask for.
- **Prefer one-liners.** Favor simple, direct solutions — comprehensions, vectorized
  pandas/numpy/astropy ops, and **reusing existing utils** — over multi-step machinery when a
  one-liner is clear and correct.
- **Match the existing code.** Mirror `ts_weatherforecast`'s level of abstraction; don't add
  layers it doesn't have.

**Steps:**

1. **Plan.** After the "Read first" step, propose: the file tree, the public API signatures,
   the artifact-loading strategy, the utils you'll reuse, and the test plan. **Stop and wait for
   my confirmation.**
2. **Implement** only after I approve the plan.
3. **Verify.** Run `pre-commit run --all-files` (using the repo's `.ts_pre_commit_config.yaml`),
   run `ruff`, and run the test suite. Fix everything until clean.
4. If you hit an ambiguity you cannot resolve from the references, ask me rather than guessing.

---

## 7. Acceptance criteria

- The notebook's forecast call runs against the new module with minimal edits and the inference
  output matches the reference pipeline within a small numerical tolerance on a shared fixture.
- `pre-commit`, `ruff`, and the full test suite pass.
- No training/fitting occurs at inference time; the package only loads artifacts and predicts.
- Docstrings, type hints, and `version_history` meet the conventions seen in
  `ts_weatherforecast`.

---

## 8. Confirm before coding (open questions)

Please answer these (or ask me) before implementing — they change the design:

1. **Slot keying (most important):** Is each per-slot Ridge corrector keyed to the **forecast
   target time** `φ_target` (the time whose temperature is being predicted — my working
   assumption in §4/§4c) or to the **issuance/"now" time** `φ_now`? This flips the slot-selection
   logic, so confirm before building.
2. **Slot set & cadence:** Is the canonical slot set exactly 12:00 → 18:30 at 30-min steps
   (equinox labels), or should it be configurable / cover more of the night? Where does the
   training script get the list?
3. **Conversion util:** The solar-time / sunrise-sunset conversion already exists in the code
   (vectorized astropy `Time`). Confirm you have found the right helper and that training and
   inference both use it, so the `φ` slots line up exactly.
4. **Skeleton contents:** `ts_weathernbeats` is a non-working initial skeleton (it has `docs/`
   and `setup.cfg`). Confirm what else is already in it so you extend the skeleton rather than
   duplicate or clobber existing files.
5. **Artifact format & location:** Confirm the shipped artifacts are the **direct-temperature,
   per-slot** models (NBEATSx trained on `y` = temperature, not `delta_T`), saved as a
   `NeuralForecast.save()` directory plus the per-slot Ridge+scaler artifacts. Where do they
   live — committed under `models/`, downloaded at runtime, or pointed to by config?
6. **EFD source:** Confirm the EFD topic/`salIndex`/field for the FeatureBuilder query (§4b) —
   is `lsst.sal.ESS.temperature` salIndex 301 `temperatureItem0` the Summit Weather Tower sensor
   the artifacts were trained on, or should it be a different sensor?
7. **Runtime integration:** Standalone library + CLI for now, or does it need to plug into the
   `ts_weatherforecast` CSC / SAL component in this iteration?
8. **Auxiliary inputs:** Should humidity/wind features (used only at long leads) be required,
   optional, or omitted in v1? (Note these need their own EFD queries beyond the temperature one.)