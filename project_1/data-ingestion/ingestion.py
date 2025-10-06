import pandas as pd
import pika
import logging, os
from ride import Ride

from datetime import datetime

QUEUE_NAME = os.getenv('QUEUE_NAME')
EXCHANGE_NAME = os.getenv('EXCHANGE_NAME')
ROUTE_KEY = os.getenv('ROUTE_KEY')
BROKER_HOST = os.getenv('BROKER_HOST')
BROKER_USER = os.getenv('BROKER_USER')
BROKER_PASS = os.getenv('BROKER_PASS')
DATAFILE = os.getenv('DATAFILE')

logging.basicConfig(
    filename='ingestion.log',
    filemode='a',
    encoding='utf-8',
    level=logging.INFO,
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M"    
)

logger = logging.getLogger(__name__)

class DataIngest:

    def __init__(self, queue_name, conn_params):
        self.queue_name = queue_name
        self.rabbitmq_conn_params = conn_params

        # Messaging connection & channel
        self.queue_connection = None
        self.channel = None
        
        self.max_retries = 5
        self.retry_delay = 5

    def is_connected(self):
        return self.queue_connection is not None

    def connect(self):
        retries = 0
        if not self.queue_connection:
            while retries < self.max_retries:
                try:
                    self.queue_connection = pika.BlockingConnection(self.rabbitmq_conn_params)
                    self.channel = self.queue_connection.channel()
                    logger.info(f"Ingestion service: Connected to RabbitMQ")
                    break
                except pika.exceptions.AMQPConnectionError as e:
                    logger.warning(f"Ingestion service: Connection failed: {e}. Retrying in {self.retry_delay} seconds...")
                    retries += 1
                    time.sleep(self.retry_delay)
            else:
                logger.error(f"Ingestion service: Failed to connect to RabbitMQ after several attempts.")

    def close_connections(self):
        if self.queue_connection:
            self.channel.stop_consuming()
            self.queue_connection.close()

    def publish_message(self, message, route_key):
        self.channel.basic_publish(
            exchange=EXCHANGE_NAME, 
            routing_key=ROUTE_KEY, 
            body=message,
            properties=pika.BasicProperties(
                headers=headers
            )
        )
        logger.debug(f"Published message: {message}")

def read_large_file(file_path, chunk_size):
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        yield chunk

def ingestion():
    try:
        conn_params = pika.ConnectionParameters(BROKER_HOST, 5672, '/', pika.PlainCredentials(BROKER_USER, BROKER_PASS))
        ingestor = DataIngest(QUEUE_NAME, conn_params)
        ingestor.connect()

        for chunk in read_large_file(DATAFILE, 1000):
            for _, row in chunk.iterrows():
                event = Ride(
                    row.ride_id, 
                    row.rideable_type, 
                    row.started_at, 
                    row.ended_at, 
                    row.start_station_id, 
                    row.end_station_id
                )

                # logger.info(f'EVENT: {event.to_json()}')
                ingestor.publish_message(event.to_json())
            # break
        ingestor.close_connections()
    except Exception as e:
        logging.error(e)

if __name__ == '__main__':
    ingestion()

