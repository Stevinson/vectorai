"""Stream the data from the local filesystem to Kafka"""
import abc
import pickle
import random
from typing import List, Union

import tensorflow as tf
from dagster import config_from_files, fs_io_manager, job, op
from google.cloud import pubsub
from google.cloud.pubsub import types as pubsub_types
from kafka import KafkaProducer

from src.helpers import pubsub_credentials
from src.settings import CONFIG_DIR, KAFKA_IP


def serialise_data(x_train, y_train) -> List[tf.Tensor]:
    x = tf.io.serialize_tensor(
        tf.image.convert_image_dtype(
            tf.convert_to_tensor(x_train, dtype=tf.float32),
            dtype=tf.float32,
            saturate=False,
        )
    )
    y = tf.io.serialize_tensor(
        tf.image.convert_image_dtype(
            tf.convert_to_tensor(y_train, dtype=tf.int8),
            dtype=tf.float32,
            saturate=False,
        )
    )
    return [x, y]


class InvalidMessengerType(Exception):
    def __init__(self, type: str):
        message = f"{type} is not one of (kafka, pubsub)"
        super().__init__(message)


class MessageProducer(abc.ABC):
    """Abstract base class for classes that send messages"""

    def __init__(self, topic: str):

        self.topic: str = topic
        self.producer: Union[KafkaMessageProducer, PubSubMessageProducer] = None

    @abc.abstractmethod
    def send(self, x, y):
        """Send data to a message broker"""
        pass

    @abc.abstractmethod
    def close(self):
        pass


class KafkaMessageProducer(MessageProducer):
    def __init__(self, topic: str):
        super().__init__(topic)
        self.producer = KafkaProducer(bootstrap_servers=[KAFKA_IP],)

    def send(self, x: bytes, y: bytes):
        self.producer.send(self.topic, key=y, value=x)

    def close(self):
        self.producer.close()


class PubSubMessageProducer(MessageProducer):
    def __init__(self, topic: str):
        super().__init__(topic)
        self.producer = pubsub.PublisherClient(
            batch_settings=pubsub_types.BatchSettings(max_messages=5, max_latency=0.1),
            credentials=pubsub_credentials(publisher=True),
        )

    def send(self, x: bytes, y: bytes):
        future = self.producer.publish(
            f"projects/vectorai-334917/topics/{self.topic}", x
        )
        # NB. Project is hard-coded
        future.result()

    def close(self):
        pass


@op(
    config_schema={"path_stream_sample": str, "topic": str, "broker": str,}
)
def generate_stream(context):
    """Stream a random subset of data to either Kafka or Google Pub/Sub"""

    # NB. This implementation only works with unlabelled data
    stream_sample = pickle.load(open(context.op_config["path_stream_sample"], "rb"))

    x_new = stream_sample[0]
    y_new = stream_sample[1]

    # Select random sample to stream
    rand = random.sample(range(0, len(x_new)), 10)

    if context.op_config["broker"] == "kafka":
        producer = KafkaMessageProducer(context.op_config["topic"])
    elif context.op_config["broker"] == "pubsub":
        producer = PubSubMessageProducer(context.op_config["topic"])
    else:
        raise InvalidMessengerType(context.op_config["broker"])

    context.log.info(
        f"Sending dataset to {context.op_config['broker']} message broker..."
    )
    for idx in rand:
        serialised_data = serialise_data(x_new[idx], y_new[idx])
        producer.send(
            tf.keras.backend.get_value(serialised_data[0]),
            tf.keras.backend.get_value(serialised_data[1]),
        )

    producer.close()


@job(
    description="Stream local data to a messaging service (Kafka or Pub/Sub)",
    resource_defs={"io_manager": fs_io_manager},
    config=config_from_files([str(CONFIG_DIR / "stream_config.yaml")]),
)
def stream_model():
    generate_stream()
