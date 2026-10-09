# Arabic Sentiment Analysis - MLOps Practitioner Project

This repository hosts a production-grade Arabic text sentiment analysis pipeline, showcasing the full journey from a Jupyter notebook to an optimized, highly scalable model-serving infrastructure.

The project is built around **MLOps best practices**—emphasizing version control, reproducibility, monitoring, clean code, and inference optimization.

## 🏗️ Project Architecture & Features Built

This project systematically applies the foundational pillars of MLOps:

1. **Modular Codebase & Dependency Management:** 
   Migrated from legacy notebooks to a structured Python project managed by `uv`, utilizing `pyproject.toml` for strict dependency locking and separating `train` dependencies (PyTorch, MLflow) from serving dependencies (ONNX Runtime, BentoML).
2. **Data & Artifact Versioning (DVC):**
   Model weights and preprocessing states are versioned natively with DVC, ensuring every deployed model traces back to the exact data snapshot it was trained on.
3. **Experiment Tracking (MLflow):**
   Integrated MLflow to log hyperparameters, metrics, and models during the training pipeline, establishing a robust Model Registry.
4. **Continuous Integration (CI):**
   Automated unit testing workflows (GitHub Actions) validating API contracts and critical logic before any merges.
5. **Load Testing (Locust):**
   Benchmarking the inference API under concurrent load to measure throughput and identify bottlenecks proactively.
6. **Observability (Prometheus & Grafana):**
   FastAPI endpoints are instrumented to expose operational metrics, ingested by Prometheus, and visualized in Grafana via a Docker Compose stack.
7. **Production Model Serving (BentoML & ONNX):**
   The application leverages BentoML to define a production-grade inference service. We explicitly exported the model to ONNX to decouple inference from PyTorch, cutting the deployment container's content size from roughly 2 GB down to ~367 MB.

---

## 📊 Project Metrics & Achievements

- **Image Size Optimization (Containerization):** By completely decoupling PyTorch and relying exclusively on ONNX runtime for inference, the Docker image footprint was drastically reduced:
  - **Base Uncompressed Disk Usage:** Reduced from >2.5 GB to **1.24 GB**
  - **Content Size (App & Dependencies):** Reduced from ~1.5 GB to **367 MB**
- **Inference Speedup:** Leveraging the ONNX Graph optimization for CPU evaluation drastically reduced p95 latency.
- **Offline Inference Checkpointing:** Checkpoints are now entirely self-contained (weights + configs), supporting zero-dependency initialization for serving.
- **Test Coverage & Reliability:** The codebase maintains strict unit test enforcement with an overall coverage of **95.5%**, exceeding the `fail-under=70` requirement, ensuring high confidence in both API and data processing logic.

---

## 📈 The Journey: Order of Features & Optimization

The repository evolved methodically, following standard MLOps maturity levels:

### Phase 1: Base Implementation & Versioning
- **Initial Migration:** Moved model logic from `ara-bert-mini` notebook to modular scripts.
- **API First:** Implemented the core REST API using FastAPI and Pydantic for rigid request validation.
- **DVC Integration:** Added `model.dvc` to version control large files out-of-band.

### Phase 2: Testing, Tracking, and Data Ingestion
- **Unit Testing & CI:** Expanded test coverage and added automated CI pipelines.
- **MLflow Tracking:** Implemented MLflow for experiment tracking, tying metrics to model training artifacts.
- **Data Pipeline:** Added automated dataset retrieval using `kagglehub` and cleaned up data dependencies.

### Phase 3: Hardware Optimization & Robustness
- **CPU PyTorch:** Refactored the environment to use PyTorch CPU wheels by default, slashing initial environment setup times and local storage constraints.
- **Self-Contained Checkpoints:** Modified the training script to export `settings.yaml` alongside `model.pt`, enabling offline inference without relying on training configuration files.
- **Load Testing (Locust):** Added a Locust suite to simulate real-world traffic against the API.

### Phase 4: Observability
- **Prometheus & Grafana:** Dockerized the monitoring stack. Instrumented the FastAPI application to emit latency and error rate metrics.

### Phase 5: Inference Optimization & Containerization (The BentoML + ONNX Era)
- **BentoML Framework:** Adopted BentoML to manage the model serving layer (`bentofile.yaml`, `@bentoml.service`).
- **ONNX Export:** Translated the heavy PyTorch model into an ONNX graph.
- **Decoupling Dependencies:** Separated `torch`, `dvc`, and `mlflow` into optional `[project.optional-dependencies]`.
- **Final Containerization:** Containerized the ONNX-backed Bento service (`uv run bentoml containerize`), producing a lean Docker image optimized for cold-start speed and resource constraints.

---

## 🚀 How to Reproduce

### 1. Environment Setup

We use `uv` for lightning-fast environment management.

```bash
# Clone the repository
git clone https://github.com/your-org/mlops_course_sentiment_analysis.git
cd mlops_course_sentiment_analysis

# Create a virtual environment and install serving dependencies
uv venv
source .venv/bin/activate
uv pip install -e .

# If you want to run training, install the optional dependencies:
uv pip install -e ".[train]"
```

### 2. Fetching Data & Artifacts
Pull the versioned data and trained model checkpoints via DVC.
```bash
dvc pull
```

### 3. Local Development (API & Monitoring)
You can launch the core FastAPI application locally.
```bash
uv run uvicorn mlops_practitioner_course.api:app --reload
```
To spin up the Prometheus and Grafana monitoring stack:
```bash
docker-compose up -d
```

### 4. Running Load Tests
With the API running, execute the Locust load tests:
```bash
locust -f locustfile.py
```

### 5. Serving with BentoML (Production Mode)
Serve the optimized ONNX model locally using BentoML's ASGI integration.
```bash
uv run bentoml serve
```

### 6. Building the Optimized Container
To containerize the service (which excludes PyTorch and runs purely on ONNX):
```bash
# 1. Build the Bento
uv run bentoml build

# 2. Containerize the Bento to an OCI-compliant Docker Image
uv run bentoml containerize sentiment-api:latest -t sentiment-api:latest

# 3. Run your optimized container!
docker run --rm -p 3000:3000 sentiment-api:latest
```
