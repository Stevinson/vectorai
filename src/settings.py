from pathlib import Path

# Filepaths

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

BUILD_DIR = BASE_DIR / "build"

DAGSTER_CONFIGS = BASE_DIR / "src" / "configs"

EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = BASE_DIR / "models"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

SRC_DIR = BASE_DIR / "src"

TFRECORDS_DATA_DIR = DATA_DIR / "tfrecords"

# Files

INITIAL_MODEL = MODELS_DIR / "mnist" / "initial_model.H5"

PUBSUB_KEY = SRC_DIR / "keys" / "vectorai-334917-1e578f1214aa.json"

STREAM_SAMPLE = EXTERNAL_DATA_DIR / "mnist" / "stream_sample.p"

TEST_SET = EXTERNAL_DATA_DIR / "mnist" / "test_set.p"

# Dagster

CONFIG_DIR = BASE_DIR / "src" / "configs"

# Message brokers

KAFKA_IP = "kafka:9094"
# KAFKA_IP = "0.0.0.0:9094"
