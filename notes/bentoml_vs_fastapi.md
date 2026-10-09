# BentoML vs FastAPI: Architecture & Benefits

This document provides a conceptual overview of why we introduced BentoML on top of our existing FastAPI setup, what problems it solves in production Machine Learning, and how the architecture is wired together.

## Why BentoML over Pure FastAPI?

FastAPI is a fantastic general-purpose web framework, but it doesn't have native primitives for Machine Learning execution. Wrapping an ML model in pure FastAPI leads to three major bottlenecks in production:

### 1. Adaptive Micro-Batching (The Biggest Win)
If 100 users hit a pure FastAPI endpoint simultaneously, FastAPI passes them to the model one-by-one (or concurrently). Deep learning models (like our BERT via ONNX) are highly inefficient at processing single items; they are optimized for processing matrices (batches).

**What BentoML adds:** 
BentoML acts as a smart queue. By adding `@bentoml.api(batchable=True, max_batch_size=64, max_latency_ms=50)` to our model service, we tell BentoML to wait up to 50ms for incoming requests. It bundles them together, passes a single matrix of up to 64 texts to our ONNX model, and then unpacks the array back to the individual HTTP responses. To build this in FastAPI, you'd need a complex Redis, Celery, and background worker loop setup.

### 2. Process Separation & Resource Isolation
In pure FastAPI, if your model takes 200ms of intense CPU/GPU compute, it blocks the Python process. Even using `async/await`, matrix multiplication holds the Global Interpreter Lock (GIL), meaning your server can't accept new TCP connections or respond to health checks during that time.

**What BentoML adds:**
It isolates the Model into its own dedicated worker process(es) (called Services/Runners). By defining `@bentoml.service(resources={"cpu": "2"})`, BentoML spins up isolated workers for the model. FastAPI just handles the lightweight network I/O, handing off the heavy lifting over shared memory to the model workers without blocking its own event loop.

### 3. Containerization & Deployment (The "Bento")
With FastAPI alone, you have to write your own `Dockerfile`, securely inject model weights into the container, and manage Python dependency versions manually.

**What BentoML adds:**
It standardizes the build process via a `bentofile.yaml`. Running `bentoml build` automatically collects the code, the exact environment, and packages it into a unified, reproducible artifact called a "Bento." You can then easily containerize it with a single command.

---

## Where Are the Updates Made?

We use a pattern called **Mounting ASGI**, which gives us the best of both worlds. We keep FastAPI for routing, data validation (Pydantic), and Prometheus metrics, but let BentoML handle the ML execution engine.

1. **`serving/model_service.py` (The Execution Engine)**: 
   This is where the pure ML logic lives. We created `ModelService` which loads the `OnnxSentimentPredictor` on startup. This is where the batching rules (`@bentoml.api`) are defined.
   
2. **`serving/service.py` (The Gateway)**: 
   This is the main entry point that BentoML runs. It tells BentoML to run our FastAPI `app` as the frontend (`@bentoml.asgi_app(app)`), and injects our `ModelService` as a dependency.

3. **`api.py` (The Frontend)**: 
   We modified our FastAPI endpoints. Instead of calling `predictor.predict_proba()` directly, the endpoints now accept the injected BentoML service and pass the data into BentoML's batching queue asynchronously:
   ```python
   probs = await model_service.to_async.predict(texts)
   ```

