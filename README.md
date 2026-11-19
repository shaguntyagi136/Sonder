# StressSense — IoT-Based Stress Monitoring System

StressSense is an IoT-based physiological stress monitoring system that collects wearable sensor data, processes it into fixed-duration windows, and uses a machine learning model to classify the user's stress state.

The project demonstrates an end-to-end pipeline using MQTT communication, real-time sensor processing, machine learning inference, and a live web dashboard.

## Features

- **Real-time sensor ingestion:** Receives BVP, EDA, temperature, and accelerometer data through MQTT.
- **Sensor windowing:** Collects sensor readings into 60-second windows.
- **Feature extraction:** Extracts 27 features for model inference.
- **Machine learning prediction:** Uses a trained Random Forest classifier to predict:
  - `BASELINE`
  - `STRESS`
- **Live dashboard:** Displays sensor activity, live signal graphs, processing progress, and prediction probabilities.
- **WebSocket communication:** Streams backend sensor, processing, and prediction events to the frontend.
- **WESAD replay:** Supports replaying recorded physiological data for end-to-end testing.

## System Architecture

```text
WESAD Dataset / Sensor Source
            |
            v
       MQTT Broker
            |
            v
       FastAPI Backend
            |
            v
    Sensor Window Manager
            |
            v
      Feature Extraction
            |
            v
    Random Forest Model
            |
            v
     Stress Prediction
            |
            v
     Live Web Dashboard
```

## Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| Communication | MQTT, WebSockets |
| MQTT Client | Paho MQTT |
| Machine Learning | Scikit-learn |
| Numerical Processing | NumPy |
| Model Serialization | Joblib |
| Frontend | HTML, CSS, JavaScript |
| Dataset | WESAD |

## Project Structure

```text
StressSense/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── mqtt_handler.py
│   │   ├── predictor.py
│   │   ├── feature_extractor.py
│   │   └── window_manager.py
│   └── tests/
├── frontend/
│   └── m1.html
├── ml/
│   ├── artifacts/
│   │   ├── metadata.json
│   │   └── stress_model.joblib
│   ├── notebook/
│   └── verify_artifacts.py
├── simulator/
│   └── wesad_replay.py
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/shaguntyagi136/Sonder.git
cd Sonder
```

### 2. Create a virtual environment

```bash
python -m venv .venv312
```

Activate it on Windows:

```powershell
.venv312\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure MQTT

Configure the MQTT broker connection and credentials using the environment variables expected by the backend MQTT configuration.

Do not commit broker credentials or other secrets to GitHub.

## Running the Application

### 1. Start the backend

From the project root:

```bash
uvicorn backend.app.main:app --reload
```

### 2. Open the dashboard

Open `frontend/m1.html` using a local development server, such as the VS Code Live Server extension.

The dashboard connects to the backend for live sensor events and predictions.

### 3. Replay WESAD data

To run the recorded-data replay through MQTT:

```bash
python simulator/wesad_replay.py --mode replay --subject S2
```

The replay requires a correctly configured and reachable MQTT broker.

For offline model evaluation:

```bash
python simulator/wesad_replay.py
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Backend and MQTT connection status |
| `GET /api/predictions/latest` | Latest model prediction |
| `WS /ws/live` | Live sensor, processing, and prediction events |

## Sensor Sampling Configuration

The system uses the following expected sampling rates for a 60-second window:

| Sensor | Sampling Rate | Expected Samples |
|---|---:|---:|
| BVP | 64 Hz | 3,840 |
| EDA | 4 Hz | 240 |
| Temperature | 4 Hz | 240 |
| Accelerometer | 32 Hz | 1,920 per axis |

## Machine Learning Model

The project uses a trained Random Forest classifier with 27 input features.

The model produces a binary classification:

- `0` — BASELINE
- `1` — STRESS

The dashboard displays the predicted class along with the model's stress and baseline probabilities.

## Testing

Run the backend test suite from the project root:

```bash
python -m pytest backend -v
```

Verify the model artifacts:

```bash
python ml/verify_artifacts.py
```

## Dataset

This project uses the **Wearable Stress and Affect Detection (WESAD)** dataset for model development and recorded-data replay.

WESAD replay is recorded physiological data, not a live measurement from a human subject. It is used to demonstrate and test the end-to-end IoT pipeline.

## Notes

- MQTT broker credentials must be configured locally.
- The trained model artifacts are included in the repository.
- The `.env` file, virtual environment, and dataset files should remain excluded from version control.
- Model predictions are experimental outputs and should not be treated as medical diagnoses.

## Author

**StressSense Project**

Repository: https://github.com/shaguntyagi136/Sonder
