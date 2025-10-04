import pika

class XYZ:

    def __init__(self, queue_name, conn_params):
        self.queue_name = queue_name
        self.rabbitmq_conn_params = conn_params

        # Messaging connection & channel
        self.queue_connection = None
        self.channel = None
        
        self.max_retries = 5
        self.retry_delay = 5

    def is_connected(self):
        return self.queue_connection is not None and self.cassandra_session is not None

    def connect_messaging(self):
        retries = 0
        if not self.queue_connection:
            while retries < self.max_retries:
                try:
                    self.queue_connection = pika.BlockingConnection(self.rabbitmq_conn_params)
                    self.channel = self.queue_connection.channel()
                    print(f"Worker node {self.node_id}: Connected to RabbitMQ", flush=True)
                    break
                except pika.exceptions.AMQPConnectionError as e:
                    print(f"Worker node {self.node_id}: Connection failed: {e}. Retrying in {self.retry_delay} seconds...")
                    retries += 1
                    time.sleep(self.retry_delay)
            else:
                print(f"Worker node {self.node_id}: Failed to connect to RabbitMQ after several attempts.")


    def consume_messages(self):
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=self.queue_name, on_message_callback=self.process_messages, auto_ack=False)
        print(f"Worker node {self.node_id}: Waiting for messages.", flush=True)
        self.channel.start_consuming()

    def close_connections(self):
        if self.queue_connection:
            self.channel.stop_consuming()
            self.queue_connection.close()

    def process_messages(self, ch, method, properties, body):
        print(f"Worker node {self.node_id}: [x] Received {body}", flush=True)
        data = body.decode('utf-8').strip().split(",")
        is_valid, error = self.check_received_data(data)
        if not is_valid:
            print(f"Worker node {self.node_id}: Invalid data received: {error}", flush=True)
            ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
        else:
            # try:
                self.message_buffer.append(data)
                if len(self.message_buffer) >= self.batch_size:
                    self.insert_into_cassandra()
                ch.basic_ack(delivery_tag=method.delivery_tag)