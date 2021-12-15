"""Integration test each of the Dagster jobs"""

import bios

from src.predict_model import predict_model
from src.settings import (
    DAGSTER_CONFIGS,
    EXTERNAL_DATA_DIR,
    INITIAL_MODEL,
    STREAM_SAMPLE,
)
from src.stream_data import stream_model
from src.train_model import train_model


def test_train():
    """Run the job that trains a model"""

    conf = bios.read(str(DAGSTER_CONFIGS / "train_config.yaml"))
    conf["ops"]["preprocess_and_split"]["config"]["data_path"] = str(
        EXTERNAL_DATA_DIR / "cifar10"
    )
    conf["ops"]["train"]["config"]["initial_model_path"] = str(INITIAL_MODEL)
    conf["ops"]["train"]["config"]["mlflow_address"] = "http://0.0.0.0:5000"

    result = train_model.execute_in_process(run_config=conf)

    assert result.success


def test_kafka_stream():
    """
    Run the job that streams data to Kafka

    NB. You need to manually set the KAFKA_IP to 0.0.0.0:9094 to test this locally
    """

    conf = bios.read(str(DAGSTER_CONFIGS / "stream_config.yaml"))
    conf["ops"]["generate_stream"]["config"]["path_stream_sample"] = str(STREAM_SAMPLE)
    conf["ops"]["generate_stream"]["config"]["broker"] = "kafka"
    # NB. You need to manually set the KAFKA_IP to 0.0.0.0:9094 to test this locally

    result = stream_model.execute_in_process(run_config=conf)

    assert result.success


def test_pubsub_stream():
    """Run the job that streams data to Kafka"""

    conf = bios.read(str(DAGSTER_CONFIGS / "stream_config.yaml"))
    conf["ops"]["generate_stream"]["config"]["path_stream_sample"] = str(STREAM_SAMPLE)
    conf["ops"]["generate_stream"]["config"]["broker"] = "pubsub"

    result = stream_model.execute_in_process(run_config=conf)

    assert result.success


def test_predict():
    """
    Test the job that uses the model to predict

    NB. You need to manually set the KAFKA_IP to 0.0.0.0:9094 to test this locally
    """

    conf = bios.read(str(DAGSTER_CONFIGS / "predict_config.yaml"))
    conf["ops"]["stream_and_predict"]["config"]["path_current_model"] = str(
        INITIAL_MODEL
    )
    conf["resources"]["db"]["config"][
        "conn_str"
    ] = "postgresql://vectoraiuser:password@localhost:5431/predictions_postgresql"

    result = predict_model.execute_in_process(run_config=conf)

    assert result.success


# test_train()
test_kafka_stream()
test_pubsub_stream()
test_predict()
