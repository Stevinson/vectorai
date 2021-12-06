"""Stream the data from the local filesystem to Kafka"""

import pickle
import random
from typing import List

import tensorflow as tf
from dagster import config_from_files, fs_io_manager, job, op
from kafka import KafkaProducer

from src.settings import CONFIG_DIR


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


@op(
    config_schema={"path_stream_sample": str, "topic": str, "kafka_ip": str,}
)
def generate_stream(context):
    """Stream dataset to Kafka"""

    context.log.info("Creating Kafka producer to send sample data")

    producer = KafkaProducer(bootstrap_servers=[f"{context.op_config['kafka_ip']}"],)

    # NB. This implementation only works with unlabelled data
    stream_sample = pickle.load(open(context.op_config["path_stream_sample"], "rb"))

    x_new = stream_sample[0]
    y_new = stream_sample[1]

    # Select random sample to stream
    rand = random.sample(range(0, len(x_new)), 100)

    context.log.info("Sending dataset to Kafka...")
    for idx in rand:
        serialised_data = serialise_data(x_new[idx], y_new[idx])
        producer.send(
            context.op_config["topic"],
            key=tf.keras.backend.get_value(serialised_data[1]),
            value=tf.keras.backend.get_value(serialised_data[0]),
        )

    producer.close()


@job(
    description="Stream local data to Kafka",
    resource_defs={"io_manager": fs_io_manager},
    config=config_from_files([str(CONFIG_DIR / "stream_config.yaml")]),
)
def stream_model():
    generate_stream()
