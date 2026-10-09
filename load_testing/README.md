# Load Testing with Locust

This directory contains configuration for load testing the Sentiment Analysis API using [Locust](https://locust.io/).

## Prerequisites

Ensure you have installed the project's dependencies, including the `dev` group which now contains `locust`. 
If you are using `uv`:

```bash
uv sync --all-groups
```

## Running the API

Before running the load tests, make sure your API is running. From the project root, you can start the API with:

```bash
uv run uvicorn mlops_practitioner_course.api:app --reload
```
*(By default, this will run on `http://127.0.0.1:8000`)*

## Running the Load Tests

You can run Locust in two modes:

### 1. Web UI Mode (Recommended)
This starts a web server where you can configure the number of users, spawn rate, and view real-time charts.

```bash
uv run locust -f load_testing/locustfile.py
```
Then open [http://localhost:8089](http://localhost:8089) in your browser. 
- Set the **Host** to `http://127.0.0.1:8000` (or wherever your API is running).
- Enter the number of concurrent users and spawn rate.
- Click "Start swarming".

### 2. Headless Mode
To run a test purely from the command line (e.g., for CI/CD or automated benchmarks) without the UI:

```bash
uv run locust -f load_testing/locustfile.py --headless -u 50 -r 10 --run-time 1m --host http://127.0.0.1:8000
```
- `-u 50`: Simulates 50 peak concurrent users.
- `-r 10`: Spawns 10 users per second.
- `--run-time 1m`: Runs the test for 1 minute before stopping automatically.

