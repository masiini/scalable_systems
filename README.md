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
   We recommend using [`uv`](https://github.com/astral-sh/uv) for fast dependency management and script running.  
   First, install `uv` (if not already installed):
   ```bash
   pip install uv
   ```
   Then install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

2. **Start the system:**  
   `docker-compose.yml`

3. **Monitor performance:**  
   Access Prometheus metrics as configured.

## Notes

- Experiments were conducted on Ubuntu 22.04 LTS under WSL2 with an AMD Ryzen 7 7840HS CPU and 7.4 GiB RAM allocated to WSL.
- Synthetic test data with planted matches is included for evaluation.
- See the project report for detailed performance results and discussion. And READMEs for data ingestion and pattern analyser.

# Project 2: GraphRAG
## Features

- **Graph Database (Kuzu):** Stores enriched Nobel laureate and mentorship data, including categories, years, institutions, and historical scholar relationships.
- **Text2Cypher with DSPy:** Translates natural language questions into Cypher queries using dynamic few-shot selection and self-repair.
- **LRU Caching:** Efficiently caches query results and schema pruning steps for fast repeated queries.
- **Self-Repair Loop:** Automatically validates and repairs generated Cypher queries using dry-run EXPLAIN.
- **Performance Benchmarking:** Measures and visualizes pipeline latency at each stage.
- **Interactive Marimo UI:** Lets users enter questions, view generated queries, and see answers in real time.

## How it Works

1. **User Input:** Enter a question in the Marimo app (e.g., "Which Physics laureates were affiliated with University of Cambridge?").
2. **Query Generation:** DSPy selects relevant exemplars and generates a Cypher query, with validation and repair if needed.
3. **Graph Query:** The query is executed on the Kuzu database, retrieving structured answers.
4. **Caching:** Results are cached for efficiency, keyed by question and schema.
5. **Answer Display:** The Marimo UI shows the Cypher query, answer, and debug info.

## Usage

1. **Install dependencies:**  
   See `requirements.txt` in the project directory.

2. **Start the Marimo UI:**  
   ```bash
   uv run marimo run graph_rag.py
   ```
   or  
   ```bash
   marimo run graph_rag.py
   ```

3. **Authentication for Embedding Models:**  
   In case you receive an error from the embedding model use, run `hf auth login` and visit https://huggingface.co/google/embeddinggemma-300m to add permissions to use the model. Also, check the boxes of "Read access to contents of all public gated repos you can access" and "Make calls to Inference Providers" from your HF Access Token Permissions

## Our Contributions

- Dynamic few-shot selection for Text2Cypher.
- LRU cache for query and schema pruning.
- Self-repair loop for Cypher validation.
- Performance benchmarking and timing breakdown.
- Integration of Kuzu, DSPy, and Marimo for a seamless graph RAG workflow.

## Data

- Nobel laureates and mentorship relationships (1901–2021), enriched with official Nobel Prize API metadata.
- Graph schema includes scholars, prizes, institutions, and mentorship edges.

## References

- [Kuzu Graph Database](https://kuzudb.com/)
- [DSPy](https://github.com/stanfordnlp/dspy)
- [Marimo](https://github.com/marimo-team/marimo)
- [Hugging Face](https://huggingface.co/)

---

For more details, see the project report and code comments.