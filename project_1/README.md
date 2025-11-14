# Citi Bike Hot Path Detection with OpenCEP

This project implements efficient pattern detection over Citi Bike data streams using the OpenCEP complex event processing (CEP) engine. The goal is to detect “hot paths”—sequences of bike trips that indicate rapid accumulation of bicycles at specific stations—while maintaining low-latency processing under bursty workloads.

## Features

- **Pattern Recognition:** Detects hot paths using a Kleene closure pattern with a 1-hour window.
- **Dynamic Load Shedding:** Maintains low latency by dropping low-utility events under high load.
- **Metrics & Monitoring:** Exposes Prometheus metrics for latency, throughput, recall, and resource usage.
- **Scalable Architecture:** Designed for extension to multiple pattern analyser replicas and persistent storage.

## Project Structure

- `pattern_recogniser/` – Pattern detection logic, shedding controller, and OpenCEP integration.
- `data-ingestion/` – Event ingestion scripts for Citi Bike data.
- `data/` – Raw and synthetic datasets, including test runs and planted matches.
- `monitoring/` – Prometheus configuration for metrics collection.
- `test_data/` – Scripts and results for synthetic test data generation and evaluation.
- `docker-compose.yml` – Container orchestration for running the system.

## Usage

1. **Install dependencies:**  
   See `requirements.txt` in each component directory.

2. **Start the system:**  
   `docker-compose.yml`

3. **Monitor performance:**  
   Access Prometheus metrics as configured.

## Notes

- Experiments were conducted on Ubuntu 22.04 LTS under WSL2 with an AMD Ryzen 7 7840HS CPU and 7.4 GiB RAM allocated to WSL.
- Synthetic test data with planted matches is included for evaluation.
- See the project report for detailed performance results and discussion. And READMEs for data ingestion and pattern analyser.