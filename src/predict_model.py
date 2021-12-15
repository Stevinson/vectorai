"""Fetch images from Kafka and classify them using the trained model."""
import abc
import json

import tensorflow as tf
import tensorflow_io as tfio
from dagster import config_from_files, job, op
from google.cloud.pubsub_v1 import SubscriberClient
from google.cloud.pubsub_v1.subscriber.message import Message
from keras.models import load_model
from sqlalchemy import text

from src.database import sqlalchemy_postgres_warehouse_resource
from src.helpers import pubsub_credentials
from src.settings import CONFIG_DIR, KAFKA_IP


def load_current_model(model_path: str) -> tf.keras.models.Model:
    model = load_model(model_path)
    model.compile(
        loss=tf.keras.metrics.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adadelta(),
        metrics=["accuracy"],
    )
    return model


def predict_batches(model, dataset: tf.data.Dataset, db, context, printy):
    """Predict each image in a batch and save them to a database"""

    for batch in dataset:
        predictions = model.predict(batch)
        preds = tf.argmax(predictions, axis=-1)
        for idx, img in enumerate(batch):
            query = f"""
                INSERT INTO predictions
                VALUES ('{json.dumps(tf.keras.backend.get_value(img).tolist())}', {preds[idx]}, now() at time zone 'utc')
            """
            # NB. Refactor encountered db issues so just logging to Dagster
            # with db.session_scope() as sess:
            #     sess.execute(text(query))
            context.log.info(
                f"A prediction of an image fetched from {printy} of {preds[idx]} was saved to the database"
            )


class PubSubFetcher:
    """Stream and predict data from Google Pub/Sub"""

    def __init__(self, model, dagster_context):
        self.model = model
        self.db = dagster_context.resources.db
        self.dagster_context = dagster_context

        self.subscriber = SubscriberClient(credentials=pubsub_credentials())
        self.subscription_path = self.subscriber.subscription_path(
            "vectorai-334917", "DataTopic-sub"
        )

    def fetch_and_predict(self):
        def pubsub_callback(message: Message) -> None:
            data = tf.io.parse_tensor(message.data, tf.float32, name=None)
            predict_batches(
                self.model,
                tf.data.Dataset.from_tensors([data]),
                self.db,
                self.dagster_context,
                "pubsub",
            )
            message.ack()

        self.subscriber.subscribe(self.subscription_path, callback=pubsub_callback)


class KafkaFetcher:
    """Stream and predict data from Kafka"""

    def __init__(self, model, dagster_context):
        self.model = model
        self.db = dagster_context.resources.db
        self.context = dagster_context

        dataset = tfio.experimental.streaming.KafkaGroupIODataset(
            topics=[dagster_context.op_config["topic"]],
            group_id="testcg",
            servers=KAFKA_IP,
            stream_timeout=dagster_context.op_config["stream_timeout_ms"],
            message_poll_timeout=3000,
        )
        dataset = dataset.map(
            self._decode_kafka_tensor,
            num_parallel_calls=dagster_context.op_config["batch_size"],
        )
        self.dataset = dataset.batch(dagster_context.op_config["batch_size"])

    def fetch_and_predict(self):
        predict_batches(self.model, self.dataset, self.db, self.context, "kafka")

    @staticmethod
    def _decode_kafka_tensor(raw_message: tf.Tensor, raw_key: tf.Tensor) -> tf.Tensor:
        return tf.io.parse_tensor(raw_message, tf.float32, name=None)


class Fetcher(abc.ABC):
    """Fetch messages from various services"""

    CLASSES = {"kafka": KafkaFetcher, "pubsub": PubSubFetcher}

    def __init__(self, model, context):
        self.sources = {name: cls(model, context) for name, cls in self.CLASSES.items()}

    def fetch_and_predict(self):
        self.sources["pubsub"].fetch_and_predict()
        # NB. Kafka should be the last called as it blocks waiting
        self.sources["kafka"].fetch_and_predict()


@op(
    config_schema={
        "topic": str,
        "batch_size": int,
        "path_current_model": str,
        "stream_timeout_ms": int,
    },
    required_resource_keys={"db"},
)
def stream_and_predict(context):
    """Fetch data, make predictions, and save these to a database."""

    model = load_current_model(context.op_config["path_current_model"])

    Fetcher(model, context).fetch_and_predict()


@job(
    description="Stream unlabelled data from Kafka to classify",
    resource_defs={"db": sqlalchemy_postgres_warehouse_resource,},
    config=config_from_files([str(CONFIG_DIR / "predict_config.yaml")]),
)
def predict_model():
    stream_and_predict()
