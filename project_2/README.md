# Graph RAG with Kuzu, DSPy and marimo

Source code for course project to build a Graph RAG with Kuzu, [DSPy](https://dspy.ai/) and [marimo](https://docs.marimo.io/) (open source, reactive notebooks for Python).

## Setup

We recommend using the `uv` package manager
to manage dependencies.

```bash
# Uses the local pyproject.toml to add dependencies
uv sync
# Or, add them manually
uv add marimo dspy kuzu polars pyarrow
# Don't forget to source virtual env
source .venv/bin/activate
```

### Start Kuzu Database
```bash
docker compose up
```
Go to `localhost:8000` you can check the UI of the database

### Create basic graph
marimo simultaneously serves three functions. You can run Python code as a script, a notebook, or as an app!

#### Run as a notebook

You can manually activate the local uv virtual environment and run marimo as follows:
```bash
# Open a marimo notebook in edit mode
marimo edit eda.py
```
Or, you can simply use uv to run marimo:
```bash
uv run marimo edit eda.py
```

#### Run as an app

To run marimo in app mode, use the `run` command.

```bash
uv run marimo run eda.py
```

#### Run as a script

Each cell block in a marimo notebook is encapsulated into functions, so you can reuse them in other
parts of your codebase. You can also run the marimo file (which is a `*.py` Python file) as you
would any other script:

```bash
uv run eda.py
```
Returns:
```
726 laureate nodes ingested
399 prize nodes ingested
739 laureate prize awards ingested
```

Depending on the stage of your project and who is consuming your code and data, each mode can be
useful in its own right. Have fun using marimo and Kuzu!

### Enrich the graph 
Create the required graph in Kuzu using the following script:

```bash
uv run create_nobel_api_graph.py
```

Alternatively, you can open/edit the script as a marimo notebook and run each cell individually to
go through the entire workflow step by step.

```bash
uv run marimo edit create_nobel_api_graph.py
```

### Run the Graph RAG pipeline as a notebook

To iterate on your ideas and experiment with your approach, you can work through the Graph RAG
notebook in the following marimo file:

```bash
uv run marimo run demo_workflow.py
```

The purpose of this file is to demonstrate the workflow in distinct stages, making it easier to
understand and modify each part of the process in marimo.

### Run the Graph RAG app

A demo app is provided in `graph_rag.py` for reference. It's very basic (just question-answering), but the
idea is general and this can be extended to include advanced retrieval workflows (vector + graph),
interactive graph visualizations via anywidget, and more. More on this in future tutorials!

```bash
uv run marimo run graph_rag.py
```
