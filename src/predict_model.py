"""Fetch images from Kafka and classify them using the trained model."""


import json
import pickle

import tensorflow as tf
import tensorflow_io as tfio
from dagster import config_from_files, job, op
from keras.models import load_model
from sqlalchemy import text

from src.database import sqlalchemy_postgres_warehouse_resource
from src.settings import CONFIG_DIR


def load_current_model(model_path: str) -> tf.keras.models.Model:
    model = load_model(model_path)
    model.compile(
        loss=tf.keras.metrics.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adadelta(),
        metrics=["accuracy"],
    )
    return model


@op(
    config_schema={
        "topic": str,
        "server": str,
        "batch_size": int,
        "path_current_model": str,
    },
    required_resource_keys={"db"},
)
def stream_and_predict(context):
    """Fetch data from Kafka, predict, and save to database."""

    dataset = tfio.experimental.streaming.KafkaGroupIODataset(
        topics=[context.op_config["topic"]],
        group_id="testcg",
        servers=context.op_config["server"],
        stream_timeout=10000,
        configuration=[
            "session.timeout.ms=7000",
            "max.poll.interval.ms=8000",
            "auto.offset.reset=earliest",
        ],
    )

    def decode_kafka_tensor(raw_message: tf.Tensor, raw_key: tf.Tensor) -> tf.Tensor:
        return tf.io.parse_tensor(raw_message, tf.float32, name=None)

    dataset = dataset.map(
        decode_kafka_tensor, num_parallel_calls=context.op_config["batch_size"]
    )
    dataset = dataset.batch(context.op_config["batch_size"])

    context.log.info("Predict the Kafka images")
    predict_batches(
        context.op_config["path_current_model"], dataset, context.resources.db, context
    )


def predict_batches(model_path: str, dataset: tf.data.Dataset, db, context):
    """Load the saved model and use it to predict the data fetched from Kafka."""

    model = load_current_model(model_path)

    for batch in dataset:
        predictions = model.predict(batch)
        preds = tf.argmax(predictions, axis=-1)
        for idx, img in enumerate(batch):
            query = f"""
                INSERT INTO predictions
                VALUES ('{json.dumps(tf.keras.backend.get_value(img).tolist())}', {preds[idx]}, now() at time zone 'utc')
            """
            with db.session_scope() as sess:
                sess.execute(text(query))


@job(
    description="Stream unlabelled data from Kafka to classify",
    resource_defs={"db": sqlalchemy_postgres_warehouse_resource,},
    config=config_from_files([str(CONFIG_DIR / "predict_config.yaml")]),
)
def predict_model():
    stream_and_predict()
