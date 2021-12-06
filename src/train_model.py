"""Train a CNN to classify categorical data"""
import pickle
from typing import Tuple

import mlflow
import tensorflow as tf
from dagster import config_from_files, fs_io_manager, job, op
from keras.callbacks import LambdaCallback
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential

from src.settings import CONFIG_DIR


def build(num_classes: int, input_shape: Tuple[int, int]):
    """Build CNN"""

    model = Sequential()
    model.add(
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape)
    )
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        loss=tf.keras.metrics.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adadelta(),
        metrics=["accuracy"],
    )
    return model


@op(
    config_schema={"data_path": str, "img_rows": int, "img_cols": int,}
)
@op
def preprocess_and_split(context):
    """
    Load raw files, preprocess them, and split into training, validation and test sets.
    """
    x_test, y_test = pickle.load(
        open(context.op_config["data_path"] + "/test_set.p", "rb")
    )
    x_train, y_train = pickle.load(
        open(context.op_config["data_path"] + "/train_set.p", "rb")
    )
    x_val, y_val = pickle.load(
        open(context.op_config["data_path"] + "/val_set.p", "rb")
    )

    image_shape = (context.op_config["img_rows"], context.op_config["img_cols"], 1)

    return x_train, y_train, x_test, y_test, x_val, y_val, image_shape


@op(
    config_schema={
        "batch_size": int,
        "epochs": int,
        "initial_model_path": str,
        "num_classes": int,
        "mlflow_address": str,
    }
)
def train(context, args):
    """Build and train the CNN."""

    x_train, y_train, x_test, y_test, x_val, y_val, input_shape = args

    num_classes = context.op_config["num_classes"]

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    y_val = tf.keras.utils.to_categorical(y_test, num_classes)

    context.log.info(
        f"Constructing CNN with images of shape: {input_shape} on"
        f"cateogorical data of {num_classes} classes."
    )
    model = build(num_classes, input_shape)

    logging_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: context.log.info(
            f"Finished epoch {epoch + 1} with {logs}"
        ),
    )

    mlflow.set_tracking_uri(context.op_config["mlflow_address"])

    with mlflow.start_run():
        model.fit(
            x_train,
            y_train,
            batch_size=context.op_config["batch_size"],
            epochs=context.op_config["epochs"],
            verbose=1,
            # validation_data=(x_val, y_val),
            callbacks=[logging_callback],
        )

        score = model.evaluate(x_test, y_test, verbose=0)
        context.log.info(f"Test - loss: {score[0]}")
        context.log.info(f"Test - accuracy: {score[1]}")

        mlflow.log_metric("Epochs", context.op_config["epochs"])
        mlflow.log_metric("Batch size", context.op_config["batch_size"])
        mlflow.log_metric("Test accuracy", score[0])
        mlflow.log_metric("Loss", score[1])

        context.log.info(f"Logging to {context.op_config['initial_model_path']}")
        model.save(context.op_config["initial_model_path"])


@job(
    description="description",
    resource_defs={"io_manager": fs_io_manager},
    config=config_from_files([str(CONFIG_DIR / "train_config.yaml")]),
)
def train_model():
    train(preprocess_and_split())
