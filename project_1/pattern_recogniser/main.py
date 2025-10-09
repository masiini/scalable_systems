import collections
import pika
import logging, os, sys, time

from datetime import datetime, timedelta

import threading
from msgspec.json import Decoder
from msgspec import Struct
from msgspec.structs import asdict

from ride import Ride, parse_datetime_hook
from shedding import (
    SheddingState, TypeChainTracker, RecentEnds,
    utility_score_type, should_forward,
    events_drop, events_fwd, matches_total, TARGET,
    latency_window
)

from CEP import CEP
from condition.Condition import Variable, SimpleCondition, BinaryCondition
from condition.CompositeCondition import AndCondition
from condition.KCCondition import KCIndexCondition, KCValueCondition
from misc.ConsumptionPolicy import ConsumptionPolicy, SelectionStrategies
from base.PatternStructure import AndOperator, SeqOperator, PrimitiveEventStructure, KleeneClosureOperator
from base.Pattern import Pattern
from base.DataFormatter import EventTypeClassifier, DataFormatter
from stream.Stream import Stream
from stream.FileStream import FileOutputStream

from prometheus_client import start_http_server, Counter, Histogram

logging.basicConfig(
    filename='cep.log',
    filemode='a',
    encoding='utf-8',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M"
)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s","%Y-%m-%d %H:%M"))
logger = logging.getLogger(__name__)
logger.addHandler(sh)

QUEUE_NAME = os.getenv('QUEUE_NAME')
EXCHANGE_NAME = os.getenv('EXCHANGE_NAME')
ROUTE_KEY = os.getenv('ROUTE_KEY')
BROKER_HOST = os.getenv('BROKER_HOST')
BROKER_USER = os.getenv('BROKER_USER')
BROKER_PASS = os.getenv('BROKER_PASS')

PREFETCH = 100
KLEENE_MIN = 2
KLEENE_MAX = 5

latency_h = Histogram('pattern_recognition_latency', 'Latency for pattern recognition')
messages_processed = Counter('pattern_recognition_messages_processed', 'Total number of messages processed for pattern recognition')

class CitibikeByRideEventTypeClassifier(EventTypeClassifier):
    def get_event_type(self, event_payload: dict):
        return event_payload['rideable_type']

class CitibikeDataFormatter(DataFormatter):
    def __init__(self, event_type_classifier: EventTypeClassifier = CitibikeByRideEventTypeClassifier()):
        super().__init__(event_type_classifier)
        self.decoder = Decoder(type=Ride, dec_hook=parse_datetime_hook)

    def parse_event(self, raw_data: str):
        evt = asdict(self.decoder.decode(raw_data))
        return evt

    def get_event_timestamp(self, event_payload: dict):
        return event_payload['ended_at']

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
        self.output_stream = FileOutputStream('/opt/app', 'patterns.txt', True)
        self.data_formatter = CitibikeDataFormatter()

        # shedding state
        self.shed = None
        self.chains = TypeChainTracker(max_gap=timedelta(minutes=45))
        self.recents = RecentEnds(ttl_sec=1800)

        self.max_retries = 5
        self.retry_delay = 5

    def is_connected(self):
        return self.queue_connection is not None

    def connect(self):
        retries = 0
        while retries < self.max_retries:
            try:
                self.queue_connection = pika.BlockingConnection(self.rabbitmq_conn_params)
                self.channel = self.queue_connection.channel()
                self.channel.queue_declare(queue=self.queue_name, durable=True)
                self.channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type='direct', durable=True)
                self.channel.queue_bind(exchange=EXCHANGE_NAME, queue=self.queue_name, routing_key=ROUTE_KEY)
                self.shed = SheddingState(self.rabbitmq_conn_params, self.queue_name)
                logger.info("Pattern node: Connected to RabbitMQ")
                return
            except pika.exceptions.AMQPConnectionError as e:
                logger.info(f"Pattern node: Connection failed: {e}. Retrying in {self.retry_delay} seconds...")
                retries += 1
                time.sleep(self.retry_delay)
        logger.info("Pattern node: Failed to connect to RabbitMQ after several attempts.")


    def close_connections(self):
        try:
            if self.shed: self.shed.stop()
        except Exception:
            pass
        if self.queue_connection:
            try: self.channel.stop_consuming()
            except Exception: pass
            try: self.queue_connection.close()
            except Exception: pass

    @latency_h.time()
    def process_messages(self, ch, method, properties, body):
        t0 = time.perf_counter()
        try:
            evt = asdict(self.data_formatter.decoder.decode(body))

            self.recents.add(evt.get("end_station_id"))

            level = self.shed.update()  # uses queue depth + local p95
            score = utility_score_type(evt, self.chains, self.recents, TARGET)

            if should_forward(level, score):
                self.input_stream.add_item(body.decode('utf-8'))
                events_fwd.inc()
            else:
                events_drop.labels(reason=f"level{level}").inc()

            messages_processed.inc()
            ch.basic_ack(delivery_tag=method.delivery_tag)
        finally:
            latency_window.add(time.perf_counter() - t0)

    def start_cep_engine(self):
        def run_engine():
            try:
                self.cep.run(self.input_stream, self.output_stream, self.data_formatter)
            except Exception:
                logger.exception("[ENGINE] error")
        threading.Thread(target=run_engine, daemon=True, name="CEPEngine").start()

    def consume_messages(self):
        self.start_cep_engine()
        self.channel.basic_qos(prefetch_count=PREFETCH)
        self.channel.basic_consume(queue=self.queue_name,
                                   on_message_callback=self.process_messages,
                                   auto_ack=False)
        logger.info(f"Pattern node: Waiting for messages. Prefetch={PREFETCH}")
        self.channel.start_consuming()

def full_pattern(event_type: str, stations: set[str]):
    """
    PATTERN SEQ (BikeTrip+ a[], BikeTrip b)
    WHERE  a[i+1].bike = a[i].bike
        AND b.end in {7,8,9}
        AND a[last].bike = b.bike
        AND a[i+1].start = a[i].end
    WITHIN 1h
    """
    consumption_policy = ConsumptionPolicy(
        primary_selection_strategy=SelectionStrategies.MATCH_ANY,
        secondary_selection_strategy=SelectionStrategies.MATCH_SINGLE,
        single=["classic_bike", "electric_bike"]
    )

    return Pattern(
        SeqOperator(
            KleeneClosureOperator(
                PrimitiveEventStructure(event_type,"a"), 
                min_size=KLEENE_MIN, 
                max_size=KLEENE_MAX
            ),
            PrimitiveEventStructure(event_type,"b")
        ),
        AndCondition(
            # Already covered by event_type
            # # (1) a[i+1].bike = a[i].bike
            # KCIndexCondition(names={"a"}, getattr_func=lambda x: x["rideable_type"],
            #                  relation_op=lambda p,n: n==p, offset=1),
            # 
            # # (3) a[last].bike = b.bike
            # KCIndexCondition(names={"a","b"}, getattr_func=lambda x: x["rideable_type"],
            #                  relation_op=lambda la, lb: la==lb, first_index=-1, second_index=0, offset=None),

            # (4) a[i+1].start = a[i].end
            KCIndexCondition(
                names={"a"},
                getattr_func=lambda x: (x["start_station_id"], x["end_station_id"]),
                relation_op=lambda prev, nxt: nxt[0] == prev[1],
                offset=1
            ),

            # (2) b.end in {7,8,9}
            SimpleCondition(
                Variable("b", lambda x: x["end_station_id"]),
                relation_op=lambda eid, S=stations: eid in S
            ),
        ),
        timedelta(hours=1),
        consumption_policy=consumption_policy
    )

# for testing
def test(event_type: str, stations: set[str]):
    return Pattern(
        SeqOperator(PrimitiveEventStructure(event_type, "b")),
        SimpleCondition(Variable("b", lambda x: x["end_station_id"]),
                        relation_op=lambda eid, S=stations: eid in S),
        timedelta(hours=1)
    )

def build_patterns(stations=TARGET):
    return [full_pattern("classic_bike", stations), full_pattern("electric_bike", stations)]

def main():
    try:
        start_http_server(8080, addr="0.0.0.0")

        patterns = build_patterns(TARGET)
        cep = CEP(patterns)

        conn_params = pika.ConnectionParameters(
            BROKER_HOST, 5672, '/',
            pika.PlainCredentials(BROKER_USER, BROKER_PASS),
            heartbeat=30, blocked_connection_timeout=300
        )
        node = Cepper(QUEUE_NAME, conn_params, cep)
        node.connect()
        node.consume_messages()

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(e, exc_info=True)


if __name__ == '__main__':
    main()
