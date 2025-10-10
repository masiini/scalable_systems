# Data Ingestion

This module handles reading Citi Bike trip data and publishing events to RabbitMQ for downstream pattern recognition.

Incoming Citi Bike trip data is parsed and validated using a custom Ride struct and datetime parsing logic (`ride.py`). 

## Usage

- Run `ingestion.py` to ingest data from CSV files and send events to the message queue.
- See `requirements.txt` for dependencies.

## Pandas vs. CSV

The ingestion script can optionally use either **pandas** or Python’s built-in **csv** module for reading input files:
- **pandas** is recommended for large datasets and provides faster, more flexible data handling.
- The **csv** module is lightweight and has no external dependencies, suitable for smaller files or minimal environments.

You can select the method in the script as needed.  
See code comments in `ingestion.py` for details.