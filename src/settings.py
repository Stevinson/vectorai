from pathlib import Path

# Filepaths

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

DAGSTER_CONFIGS = BASE_DIR / "src" / "configs"

EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = BASE_DIR / "models"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

TFRECORDS_DATA_DIR = DATA_DIR / "tfrecords"

# Files

INITIAL_MODEL = MODELS_DIR / "cifar10" / "initial_model.H5"

STREAM_SAMPLE = EXTERNAL_DATA_DIR / "cifar10" / "stream_sample.p"

TEST_SET = EXTERNAL_DATA_DIR / "cifar10" / "test_set.p"

# Dagster

CONFIG_DIR = BASE_DIR / "src" / "configs"
