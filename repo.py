from dagster import repository

from src.predict_model import predict_model
from src.stream_data import stream_model
from src.train_model import train_model


@repository
def repository():
    return [train_model, stream_model, predict_model]
