"""Script to download datasets"""
import pickle

import numpy as np
import tensorflow as tf
from keras.datasets import mnist

from src.settings import EXTERNAL_DATA_DIR


def download_mnist():

    img_rows, img_cols = 28, 28

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_val, y_train, y_val = (
        x_train[:50000],
        x_train[50000:],
        y_train[:50000],
        y_train[50000:],
    )

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_val = x_val.astype("float32")

    x_train /= 255
    x_test /= 255
    x_val /= 255

    x_stream = x_train[:2000]
    y_stream = y_train[:2000]
    stream_sample = [x_stream, y_stream]

    test_set = [x_test, y_test]
    train_set = [x_train, y_train]
    val_set = [x_val, y_val]

    pickle.dump(
        stream_sample, open(str(EXTERNAL_DATA_DIR / "mnist/stream_sample.p"), "wb"),
    )
    pickle.dump(
        test_set, open(str(EXTERNAL_DATA_DIR / "mnist/test_set.p"), "wb"),
    )
    pickle.dump(
        train_set, open(str(EXTERNAL_DATA_DIR / "mnist/train_set.p"), "wb"),
    )
    pickle.dump(
        val_set, open(str(EXTERNAL_DATA_DIR / "mnist/val_set.p"), "wb"),
    )


def download_cifar10_fashion():

    img_rows, img_cols = 32, 32

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = np.dot(x_train, [0.299, 0.587, 0.114])
    x_test = np.dot(x_test, [0.299, 0.587, 0.114])

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train, x_val, y_train, y_val = (
        x_train[:25000],
        x_train[25000:],
        y_train[:40000],
        y_train[40000:],
    )

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_val = x_val.astype("float32")

    x_train /= 255
    x_test /= 255
    x_val /= 255

    x_stream = x_train[:2000]
    y_stream = y_train[:2000]
    stream_sample = [x_stream, y_stream]

    test_set = [x_test, y_test]
    train_set = [x_train, y_train]
    val_set = [x_val, y_val]

    pickle.dump(
        stream_sample, open(str(EXTERNAL_DATA_DIR / "cifar10/stream_sample.p"), "wb"),
    )
    pickle.dump(
        test_set, open(str(EXTERNAL_DATA_DIR / "cifar10/test_set.p"), "wb"),
    )
    pickle.dump(
        train_set, open(str(EXTERNAL_DATA_DIR / "cifar10/train_set.p"), "wb"),
    )
    pickle.dump(
        val_set, open(str(EXTERNAL_DATA_DIR / "cifar10/val_set.p"), "wb"),
    )


download_mnist()
# download_cifar10_fashion()
