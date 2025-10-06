import pika
import logging, os, sys

from datetime import timedelta

from ride import Ride
from CEP import CEP
from condition.Condition import Variable, TrueCondition, BinaryCondition, SimpleCondition
from condition.CompositeCondition import AndCondition
from condition.BaseRelationCondition import EqCondition, GreaterThanCondition, GreaterThanEqCondition, \
    SmallerThanEqCondition
from base.PatternStructure import AndOperator, SeqOperator, PrimitiveEventStructure
from base.Pattern import Pattern


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

class Cepper:

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
                    logger.info(f"Pattern node: Connected to RabbitMQ")
                    break
                except pika.exceptions.AMQPConnectionError as e:
                    logger.info(f"Pattern node: Connection failed: {e}. Retrying in {self.retry_delay} seconds...")
                    retries += 1
                    time.sleep(self.retry_delay)
            else:
                logger.info(f"Pattern node: Failed to connect to RabbitMQ after several attempts.")


    def consume_messages(self):
        self.channel.basic_qos(prefetch_count=1000)
        self.channel.basic_consume(queue=self.queue_name, on_message_callback=self.process_messages, auto_ack=False)
        logger.info(f"Pattern node: Waiting for messages.")
        self.channel.start_consuming()

    def close_connections(self):
        if self.queue_connection:
            self.channel.stop_consuming()
            self.queue_connection.close()

    def process_messages(self, ch, method, properties, body):
        logger.info(f"Pattern node: [x] Received {body.decode('utf-8')}")
        data = body.decode('utf-8').strip().split(",")

        ch.basic_ack(delivery_tag=method.delivery_tag)


def main():
    try:
        pattern = Pattern(
            SeqOperator(PrimitiveEventStructure("GOOG", "a"), 
                        PrimitiveEventStructure("GOOG", "b"), 
                        PrimitiveEventStructure("GOOG", "c")),
            SimpleCondition(Variable("a", lambda x: x["Peak Price"]), 
                            Variable("b", lambda x: x["Peak Price"]),
                            Variable("c", lambda x: x["Peak Price"]),
                            relation_op=lambda x,y,z: x < y < z),
            timedelta(minutes=3)
        )
        cep = CEP([pattern])
        logger.info(f'Testing: {cep}')
        conn_params = pika.ConnectionParameters(BROKER_HOST, 5672, '/', pika.PlainCredentials(BROKER_USER, BROKER_PASS))
        pattern_detector = Cepper(QUEUE_NAME, conn_params)
        pattern_detector.connect()
        # pattern_detector.consume_messages()

        pattern_detector.close_connections()

    except Exception as e:
        logger.error(e)    


if __name__ == '__main__':
    main()