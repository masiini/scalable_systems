import pika
import logging, os, sys, time

from datetime import datetime, timedelta

from msgspec.json import Decoder
from msgspec import Struct
from msgspec.structs import asdict

from ride import Ride, parse_datetime_hook

from CEP import CEP
from condition.Condition import Variable, SimpleCondition
from condition.CompositeCondition import AndCondition
from condition.KCCondition import KCIndexCondition, KCValueCondition
from base.PatternStructure import AndOperator, SeqOperator, PrimitiveEventStructure, KleeneClosureOperator
from base.Pattern import Pattern
from base.DataFormatter import EventTypeClassifier, DataFormatter
from stream.FileStream import FileOutputStream
from stream.Stream import Stream


logging.basicConfig(
    filename='cep.log',
    filemode='a',
    encoding='utf-8',
    level=logging.INFO,
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M"    
)

logger = logging.getLogger(__name__)

QUEUE_NAME = os.getenv('QUEUE_NAME')
EXCHANGE_NAME = os.getenv('EXCHANGE_NAME')
ROUTE_KEY = os.getenv('ROUTE_KEY')
BROKER_HOST = os.getenv('BROKER_HOST')
BROKER_USER = os.getenv('BROKER_USER')
BROKER_PASS = os.getenv('BROKER_PASS')

class CitibikeByRideEventTypeClassifier(EventTypeClassifier):
    def get_event_type(self, event_payload: dict):
        return event_payload['ride_id']

class CitibikeDataFormatter(DataFormatter):
    def __init__(self, event_type_classifier: EventTypeClassifier = CitibikeByRideEventTypeClassifier()):
        super().__init__(event_type_classifier)
        self.decoder = Decoder(type=Ride, dec_hook=parse_datetime_hook)

    def parse_event(self, raw_data: str):
        return asdict(self.decoder.decode(raw_data))

    def get_event_timestamp(self, event_payload: dict):
        return event_payload['started_at']

class Cepper:

    def __init__(self, queue_name, conn_params, cep):
        self.queue_name = queue_name
        self.rabbitmq_conn_params = conn_params

        # Messaging connection & channel
        self.queue_connection = None
        self.channel = None
        
        # CEP
        logger.info(f'Initializing CEP')
        self.cep = cep
        self.input_stream = Stream()
        self.output_stream = FileOutputStream('/opt/app', 'patterns.txt')
        self.data_formatter = CitibikeDataFormatter()

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
                    logger.info(f"Pattern node: Connected to RabbitMQ")
                    break
                except pika.exceptions.AMQPConnectionError as e:
                    logger.info(f"Pattern node: Connection failed: {e}. Retrying in {self.retry_delay} seconds...")
                    retries += 1
                    time.sleep(self.retry_delay)
            else:
                logger.info(f"Pattern node: Failed to connect to RabbitMQ after several attempts.")


    def consume_messages(self):
        import threading
        threading.Thread(target=self.start_cep_engine, daemon=True).start()

        self.channel.basic_qos(prefetch_count=5000)
        self.channel.basic_consume(queue=self.queue_name, on_message_callback=self.process_messages, auto_ack=False)
        logger.info(f"Pattern node: Waiting for messages.")
        self.channel.start_consuming()

    def start_cep_engine(self):
        logger.info("Starting CEP engine...")
        self.cep.run(self.input_stream, self.output_stream, self.data_formatter)


    def close_connections(self):
        if self.queue_connection:
            self.channel.stop_consuming()
            self.queue_connection.close()

    def process_messages(self, ch, method, properties, body):
        data = body.decode('utf-8').strip()
        self.input_stream.add_item(data)
        ch.basic_ack(delivery_tag=method.delivery_tag)

def main():
    try:
        pattern = Pattern(
            SeqOperator(
                KleeneClosureOperator(
                    PrimitiveEventStructure("BikeTrip", "a"), 
                    min_size=1
                ),
                PrimitiveEventStructure("BikeTrip", "b")
            ),
            AndCondition(
                # a[i+1].bike == a[i].bike
                KCIndexCondition(
                    names = {"a"},
                    getattr_func = lambda x: x.rideable_type,
                    relation_op = lambda bike_a, bike_a1: bike_a == bike_a1,
                    offset = 1
                ),
                # b-end in (7,8,9)
                SimpleCondition(
                    Variable("b", lambda x: x.end_station_id),
                    relation_op = lambda end_station: end_station in (7, 8, 9)
                ),
                # a[last].bike = b.bike
                # KCIndexCondition(
                #     names = {"a", "b"},
                #     getattr_func = lambda x: x.rideable_type,
                #     relation_op = lambda bike_a, bike_b: bike_a == bike_b,
                #     index = -1
                # ),
                # a[i+1].start = a[i].end
                KCIndexCondition(
                    names = {"a"},
                    getattr_func = lambda x: (x.start_station_id, x.end_station_id),
                    relation_op = lambda bike_a, bike_a1: bike_a1[0] == bike_a[1],
                    offset = 1
                )
            ),
            timedelta(hours=1)
        )
        cep = CEP([pattern])
        conn_params = pika.ConnectionParameters(BROKER_HOST, 5672, '/', pika.PlainCredentials(BROKER_USER, BROKER_PASS))
        pattern_detector = Cepper(QUEUE_NAME, conn_params, cep)
        pattern_detector.connect()
        pattern_detector.consume_messages()


    except Exception as e:
        logger.error(e, exc_info=True)


if __name__ == '__main__':
    main()