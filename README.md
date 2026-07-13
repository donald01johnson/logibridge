# LogiBridge

An end-to-end Edge AI system for real-time cold-chain fleet monitoring.

## Project Overview

LogiBridge is developed for the **AIML ZG535 Edge AI Systems** assignment.  
The project models a cold-chain logistics deployment involving refrigerated
trucks carrying temperature-sensitive cargo.

The system collects temperature, vibration, and door-state sensor data,
processes the data locally, and classifies the cargo condition into one of
three operational states:

| Class | State | Description |
|---:|---|---|
| 0 | Normal | Cargo and vehicle conditions are within acceptable limits |
| 1 | Warning | Early deviation or borderline operating condition detected |
| 2 | Critical | Serious threshold breach requiring immediate attention |

The complete inference pipeline is designed to operate locally at the edge,
including during periods of unreliable or unavailable network connectivity.

---

## Project Objectives

The project will:

- Simulate temperature, vibration, and door-event sensor streams.
- Publish sensor readings using MQTT.
- Apply moving-average filtering and sliding-window preprocessing.
- Extract a six-feature fused sensor vector.
- Generate labelled Normal, Warning, and Critical datasets.
- Train a three-class multilayer perceptron classifier.
- Containerise the inference service using Docker.
- Monitor input drift using Population Stability Index.
- Automate edge deployment using Ansible.
- Create post-training quantised and pruned model variants.
- Benchmark model accuracy, latency, size, recall, and energy consumption.
- Recommend a suitable model and hardware configuration for deployment.

---

## System Classification

| Label | Cargo State | Expected Interpretation |
|---:|---|---|
| 0 | Normal | Temperature and vibration remain within normal operating ranges |
| 1 | Warning | One or more measurements show moderate deviation |
| 2 | Critical | A severe anomaly or unsafe combined condition is detected |

The final deployed model must prioritise reliable detection of the Critical
class because missed critical conditions may result in cargo damage.

---

## Repository Structure

    logibridge/
    ├── README.md
    ├── scenario_architecture/
    │   ├── constraint_analysis.md
    │   └── system_architecture.png
    ├── hardware/
    │   └── hardware_justification.md
    ├── data_pipeline/
    │   ├── simulator.py
    │   ├── preprocessing.py
    │   ├── training_stats.npy
    │   └── mqtt_architecture.md
    ├── training/
    │   ├── generate_dataset.py
    │   ├── train_model.py
    │   ├── convert_ptq.py
    │   ├── prune_quantise.py
    │   └── models/
    ├── inference/
    │   ├── Dockerfile
    │   ├── inference_service.py
    │   └── model.tflite
    ├── monitoring/
    │   ├── drift_monitor.py
    │   └── reference_dist.json
    ├── deployment/
    │   └── logibridge_deploy.yml
    ├── optimisation/
    │   ├── benchmark.py
    │   └── results/
    │       ├── benchmark_results.csv
    │       └── pareto_chart.png
    └── reports/
        ├── phase1_report.pdf
        ├── phase2_report.pdf
        └── final_report.pdf

---

## Repository Contents

### `scenario_architecture/`

Contains the deployment-context analysis and system architecture.

- `constraint_analysis.md` — analysis of latency, bandwidth, connectivity,
  privacy, offline operation, and deployment constraints.
- `system_architecture.png` — complete edge-system architecture diagram.

### `hardware/`

Contains hardware comparison and deployment-platform justification.

- `hardware_justification.md` — constraint-triangle comparison, arithmetic
  intensity, Roofline analysis, and final hardware recommendation.

### `data_pipeline/`

Contains sensor simulation, preprocessing, feature extraction, and MQTT design.

- `simulator.py` — generates temperature, vibration, and door-event data.
- `preprocessing.py` — performs filtering, windowing, feature extraction, and
  normalisation.
- `training_stats.npy` — stores training-set statistics used for
  normalisation.
- `mqtt_architecture.md` — documents MQTT topics, QoS decisions, publishers,
  subscribers, offline behaviour, and data-fusion rationale.

### `training/`

Contains dataset generation, model training, quantisation, and pruning scripts.

- `generate_dataset.py` — generates labelled windows for the three classes.
- `train_model.py` — trains and evaluates the baseline MLP classifier.
- `convert_ptq.py` — converts the baseline model using INT8 post-training
  quantisation.
- `prune_quantise.py` — applies pruning and INT8 quantisation.
- `models/` — stores generated baseline and optimised model artifacts.

### `inference/`

Contains the deployable edge-inference service.

- `Dockerfile` — defines the inference container.
- `inference_service.py` — subscribes to MQTT sensor features, runs inference,
  and publishes predictions.
- `model.tflite` — deployment-ready TensorFlow Lite model.

### `monitoring/`

Contains input-distribution drift monitoring.

- `drift_monitor.py` — calculates Population Stability Index and raises drift
  alerts.
- `reference_dist.json` — contains the clean reference feature distribution.

### `deployment/`

Contains deployment automation.

- `logibridge_deploy.yml` — Ansible playbook for deploying the model and
  inference container to an edge node.

### `optimisation/`

Contains benchmarking and optimisation results.

- `benchmark.py` — benchmarks all model variants.
- `results/benchmark_results.csv` — stores measured benchmark values.
- `results/pareto_chart.png` — visualises model trade-offs.

### `reports/`

Contains assignment reports.

- `phase1_report.pdf` — Phase 1 report.
- `phase2_report.pdf` — Phase 2 report.
- `final_report.pdf` — consolidated final report.

---

## Assignment Component Mapping

| Component | Scope | Main Artifacts |
|---|---|---|
| A | Constraint analysis and system architecture | `scenario_architecture/` |
| B | Hardware selection and Roofline analysis | `hardware/` |
| C | Sensor simulation, MQTT, preprocessing, and fusion | `data_pipeline/` |
| D | Dataset generation, model training, and Docker inference | `training/`, `inference/` |
| E | Drift monitoring, Ansible deployment, and OTA strategy | `monitoring/`, `deployment/` |
| F | Quantisation, pruning, benchmarking, and recommendation | `training/`, `optimisation/` |

---

## Planned Data Pipeline

    Sensor Simulator
           |
           v
      MQTT Broker
           |
           v
    Moving-Average Filter
           |
           v
    Sliding-Window Processor
           |
           v
    Six-Feature Sensor Vector
           |
           v
       Normalisation
           |
           v
       Edge ML Model
           |
           v
    MQTT Inference Result
           |
           +------> Local Alert Log
           |
           +------> Drift Monitor
           |
           +------> Operations Uplink

---

## Sensor Inputs

### Temperature

- Represents refrigerated cargo-compartment temperature.
- Produced at the sampling rate specified in the assignment.
- Supports normal behaviour and temperature-drift anomalies.

### Vibration

- Represents vehicle or refrigeration-unit vibration.
- Supports normal behaviour and abnormal-vibration injection.

### Door Events

- Represents discrete cargo-door OPEN and CLOSE events.
- Provides operational context for temperature and vibration changes.

---

## Preprocessing Pipeline

The preprocessing pipeline will include:

1. Sensor-data validation.
2. Five-sample moving-average filtering.
3. Thirty-second sliding windows.
4. Ten-second window step.
5. Six-feature extraction.
6. Feature-level sensor fusion.
7. Normalisation using saved training statistics.
8. Comparison using correct and deliberately shifted statistics.

The exact feature definitions and implementation will be documented in
`data_pipeline/preprocessing.py` and `data_pipeline/mqtt_architecture.md`.

---

## Model Architecture

The baseline classifier will use a multilayer perceptron with:

- Six input features.
- First hidden layer with 32 units and ReLU activation.
- Second hidden layer with 16 units and ReLU activation.
- Three-class output layer.
- Normal, Warning, and Critical output classes.

The baseline and optimised models will be evaluated using the assignment's
required accuracy, recall, latency, file-size, and energy criteria.

---

## Model Variants

| Model | Description |
|---|---|
| M1 | FP32 baseline model |
| M2 | INT8 post-training quantised model |
| M3 | Pruned and INT8-quantised model |

The final deployment recommendation will be based on measured results rather
than assumed performance.

---

## Benchmark Metrics

Each model variant will be evaluated using:

- Mean inference latency.
- 95th-percentile inference latency.
- Model file size.
- Overall classification accuracy.
- Critical-class recall.
- Estimated energy used per inference.

Measured values will be stored in:

    optimisation/results/benchmark_results.csv

The model trade-off visualisation will be stored in:

    optimisation/results/pareto_chart.png

---

## Edge MLOps

The project includes:

- Population Stability Index drift monitoring.
- Rolling inference-window analysis.
- Reference-distribution comparison.
- Drift-alert generation.
- Model and container deployment using Ansible.
- Idempotency verification of the Ansible playbook.
- Evaluation of full-replacement, canary, and shadow OTA strategies.

---

## Environment

The development environment uses:

- Ubuntu Linux.
- Python 3.11.
- TensorFlow.
- TensorFlow Model Optimization Toolkit.
- Eclipse Mosquitto.
- Paho MQTT.
- Docker.
- Ansible.
- Git and GitHub.

---

## Local Environment Setup

Create or activate the Python virtual environment before installing project
dependencies.

    python3.11 -m venv logibridge
    source logibridge/bin/activate
    python -m pip install --upgrade pip

Install the project dependencies after `requirements.txt` is created:

    python -m pip install -r requirements.txt

Verify Python:

    python --version

Start and verify Mosquitto:

    sudo systemctl start mosquitto
    sudo systemctl status mosquitto

Verify Docker:

    docker --version
    docker run --rm hello-world

---

## Development Workflow

The planned implementation sequence is:

1. Create and verify the repository structure.
2. Implement the sensor simulator.
3. Implement preprocessing and feature extraction.
4. Document MQTT architecture and data fusion.
5. Generate the labelled dataset.
6. Train and evaluate the baseline model.
7. Implement Docker-based inference.
8. Implement PSI drift monitoring.
9. Implement the Ansible deployment playbook.
10. Generate quantised and pruned model variants.
11. Benchmark all model variants.
12. Complete architecture and hardware documentation.
13. Produce phase reports and the final report.
14. Record the final demonstration.

---

## Running the Project

The commands in this section will become active as each corresponding script
is implemented.

Generate the dataset:

    python training/generate_dataset.py

Train the baseline model:

    python training/train_model.py

Run the simulator in normal mode:

    python data_pipeline/simulator.py --anomaly none

Run a temperature-drift scenario:

    python data_pipeline/simulator.py --anomaly temp_drift

Run a vibration-anomaly scenario:

    python data_pipeline/simulator.py --anomaly vibration

Run a combined-anomaly scenario:

    python data_pipeline/simulator.py --anomaly combined

Run model benchmarking:

    python optimisation/benchmark.py

Run the Ansible deployment:

    ansible-playbook deployment/logibridge_deploy.yml

These commands are documented here as the intended interface. They will be
validated as implementation progresses.

---

## Generated Artifacts

The following files are generated during project execution and should not be
manually edited:

- `data_pipeline/training_stats.npy`
- Models stored under `training/models/`
- `inference/model.tflite`
- `monitoring/reference_dist.json`
- `optimisation/results/benchmark_results.csv`
- `optimisation/results/pareto_chart.png`
- Report PDF files under `reports/`

---

## Project Status

| Area | Status |
|---|---|
| Environment setup | In progress |
| Repository structure | In progress |
| Sensor simulator | Pending |
| Preprocessing pipeline | Pending |
| MQTT documentation | Pending |
| Dataset generation | Pending |
| Model training | Pending |
| Docker inference | Pending |
| Drift monitoring | Pending |
| Ansible deployment | Pending |
| Model optimisation | Pending |
| Benchmarking | Pending |
| Architecture documentation | Pending |
| Hardware justification | Pending |
| Reports | Pending |
| Demonstration video | Pending |

---

## Reproducibility

The repository will contain:

- Source code required to generate the dataset.
- Source code required to train all models.
- Saved preprocessing statistics.
- Model-conversion and optimisation scripts.
- Deployment configuration.
- Benchmark scripts and results.
- Architecture and design documentation.
- Assignment reports.

The local Python virtual environment is intentionally excluded from Git.
Dependencies will be recreated using `requirements.txt`.

---

## Author

**Donald Johnson A.**

---

## License

This repository is created for academic assignment work. Redistribution and
reuse should comply with the applicable course and institutional policies.
