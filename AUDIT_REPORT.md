# Phase 1 Backend Audit and Repair

## Scope

Audited only the requested Python backend modules, their existing tests,
`requirements.txt`, and model metadata. No frontend, simulator, WESAD data,
`.env`, or model artifact was changed.

## Problems Found and Root Causes

| Status | Area | Finding and root cause |
|---|---|---|
| VERIFIED | `backend/app/window_manager.py` | The reported `IndentationError` was caused by file corruption, not a single extra indent. The file began with an orphaned `) -> bool:` method fragment and retained only the final 109 lines of a class method body. Its imports, constants, data classes, `WindowManager` class declaration, ingestion path, and timestamp parser were missing. |
| VERIFIED | Window integration | `mqtt_handler.py` imports `MODALITIES`, `WindowManager`, and `parse_timestamp`, while tests require `EXPECTED_COUNTS`, async processing, device/window separation, duplicate rejection, timestamp ordering, and expiry behavior. These required interfaces were absent because of the corrupted module. |
| VERIFIED | Startup behavior | `MQTTHandler.start()` used Paho's blocking `connect()` in the FastAPI lifespan. An unreachable broker could delay or prevent Uvicorn readiness, even though `/health` is designed to expose degraded status. |
| VERIFIED | Dependencies | `.venv312` imports FastAPI, Paho MQTT, NumPy, scikit-learn, and joblib successfully; `pip check` reports no broken requirements. Existing requirements cover the backend runtime and test dependencies. |
| VERIFIED | Model metadata | Metadata declares 27 features and matches the trained model's `n_features_in_` and classes. No artifact or metadata change was required. |

## Files Changed

- `backend/app/window_manager.py`
  - Reconstructed the missing module around the surviving implementation.
  - Restored timestamp parsing, sample/window models, 60-second buffering,
    validation, expiry, progress/status callbacks, ordered feature input, and
    concurrent processing.
  - Preserved the existing feature extractor/predictor pipeline and MQTT-facing
    interface.
- `backend/app/mqtt_handler.py`
  - Replaced blocking Paho `connect()` with `connect_async()` before
    `loop_start()`, allowing the API to start while MQTT connects/reconnects in
    the network thread.
- `AUDIT_REPORT.md`
  - Replaced the prior unrelated audit content with this Phase 1 report.

## Validation Results

| Check | Status | Result |
|---|---|---|
| Python environment | VERIFIED | Used `.venv312` (Python 3.12.5). |
| Dependency integrity | VERIFIED | `.venv312\\Scripts\\python.exe -m pip check` returned `No broken requirements found.` |
| Compilation | VERIFIED | `.venv312\\Scripts\\python.exe -m compileall -q backend\\app` completed successfully. |
| Existing tests | VERIFIED | `.venv312\\Scripts\\python.exe -m pytest backend -q` returned `16 passed`. |
| Model/artifact validation | VERIFIED | `ml\\verify_artifacts.py` passed: RandomForestClassifier, 27 input features, classes `[0, 1]`. |
| Port pre-check | VERIFIED | Port 8000 was free before the smoke test; no duplicate server was launched. |
| FastAPI startup and health | VERIFIED | Uvicorn was started briefly on `127.0.0.1:8000`; `GET /health` returned HTTP 200 with `{"status":"degraded","mqtt_connected":false,"mqtt_error":null}`. |

## Remaining Issues

- VERIFIED: The API deliberately reports `degraded` until a configured MQTT
  broker connects. Live broker connectivity was not asserted because it depends
  on external broker availability and credentials; no credentials were read
  into the report or exposed.
- NOT VERIFIED: No real MQTT publish/subscribe exchange was run against the
  external broker. Message validation and the backend pipeline are covered by
  the existing unit tests.
- VERIFIED: No changes were made outside Phase 1 backend repair.

---

# Phase 2 Sensor Pipeline

## Inspection

- VERIFIED: Reviewed `backend/app/window_manager.py`,
  `backend/app/mqtt_handler.py`, `backend/app/feature_extractor.py`, and all
  existing backend tests before changing the pipeline.
- VERIFIED: Inspected the WESAD E4 CSV layout without modifying it. `BVP.csv`,
  `EDA.csv`, and `TEMP.csv` begin with a start timestamp then their rate
  (64/4/4 Hz). `ACC.csv` provides timestamp/rate triples and XYZ vectors at
  32 Hz. This matches the established canonical pipeline rates.

## Changes

- `backend/app/mqtt_handler.py`
  - Documented one canonical JSON MQTT message schema, topic layout, fields,
    rates, and the mapping to WESAD BVP/EDA/TEMP/ACC.
  - Added bounded device-ID validation (`A-Z`, `a-z`, `0-9`, `.`, `_`, `-`).
  - Retained finite numeric validation for scalar and XYZ readings; explicitly
    reject boolean sample rates.
- `backend/app/window_manager.py`
  - Added configurable positive integer `window_seconds` (default: 60) and
    derives exact modality counts from the configured duration and fixed rates.
  - Preserves separate device/window/modality buffers, timestamp ordering,
    duplicate rejection, gap checks, and explicit incomplete-window expiry.
  - Rejects non-finite and boolean values even if callers bypass MQTT schema
    validation.
  - Emits `WINDOW_COMPLETED` once and optionally hands off a completed payload
    exactly once through `window_callback`. Its `reading` value has the exact
    `feature_extractor.py` contract: `bvp`, `eda`, `skin_temperature`, and
    `acc_x`/`acc_y`/`acc_z` arrays.
- `backend/test_sensor_pipeline.py`
  - Added focused, credential-independent Phase 2 coverage.

## Requirement Status

| Requirement | Status | Evidence |
|---|---|---|
| Consistent documented schema | VERIFIED | Canonical schema is documented beside `SENSOR_TOPIC`; topic and payload must agree. |
| BVP, EDA, TEMP, and ACC XYZ support | VERIFIED | Canonical modalities and 64/4/4/32 Hz rates are validated; focused tests cover all four. |
| Device, modality, timestamp, and value validation | VERIFIED | Topic, bounded device ID, timezone-aware ISO-8601 timestamp, sequence, rate, and finite scalar/XYZ values are checked. |
| Malformed, duplicate, and out-of-order handling | VERIFIED | Invalid messages raise/reject; duplicates are rejected; timestamp-consistent out-of-order samples are sorted before handoff. |
| Independent device/modality buffers | VERIFIED | Buffers are keyed by device then timestamp window, with separate modality lists; covered by tests. |
| Configurable timestamp-based 60-second windows | VERIFIED | Default is 60 seconds; positive `window_seconds` derives expected counts from actual timestamps and rates. |
| Exact counts and no fabricated missing samples | VERIFIED | Completion requires 3,840 BVP, 240 EDA, 240 TEMP, and 1,920 ACC vectors; short/gapped windows emit `WINDOW_INCOMPLETE` and are retired. |
| Exactly-once feature-extractor handoff | VERIFIED | Retired-window keys prevent repeats; focused full-window test observes one callback with a feature-extractor-compatible reading. |
| Live MQTT integration | NOT VERIFIED | Tests deliberately do not use HiveMQ credentials or an external broker. |
| Full WESAD replay validation | NOT VERIFIED | The on-disk WESAD format was inspected only. No synthetic unit test is presented as evidence of a real WESAD replay. |

## Validation Results

| Command | Status | Result |
|---|---|---|
| `.venv312\\Scripts\\python.exe -m pytest backend\\test_sensor_pipeline.py -q` | VERIFIED | `5 passed in 4.17s` |
| `.venv312\\Scripts\\python.exe -m pytest backend -q` | VERIFIED | `21 passed in 23.12s` |
| `.venv312\\Scripts\\python.exe -m compileall -q backend\\app backend\\test_sensor_pipeline.py` | VERIFIED | Completed successfully (`COMPILE_OK`). |
| `.venv312\\Scripts\\python.exe ml\\verify_artifacts.py` | VERIFIED | Existing model and metadata remain compatible: 27 features, classes `[0, 1]`. |

## Remaining Limitations

- NOT VERIFIED: A live HiveMQ broker exchange was not run; passing unit tests
  does not prove deployment credentials, TLS configuration, broker routing, or
  reconnection behavior.
- NOT VERIFIED: WESAD files were inspected but not replayed as live input in
  this phase. The dataset and model artifacts remain unchanged.
- VERIFIED: No frontend, simulator, `.env`, WESAD data, or ML artifact was
  modified. Phase 3 was not started.
