import json
from typing import Any

from flask import Flask, request, jsonify, Response
import pandas as pd

from dream.data_models.request import TrainRequest, PredictRequest
from dream.model.data_processing import preprocess_data, load_data
from dream.model.model_inference import predict
from dream.model.model_training import train_model


def load_from_json(json_path: str) -> Any:
    with open(json_path) as file:
        return json.load(file)


app = Flask(__name__)


@app.route("/")
def home():
    return "<h1>Dream home assessment</h1>"


@app.route("/train", methods=["POST"])
def train_route() -> str:
    data = request.json
    train_request = TrainRequest(**data)
    df = load_data(train_request.dataset_path)
    preprocessed_data = preprocess_data(df)
    trained_model_path = train_model(preprocessed_data, train_request.model_type)
    print(f"The trained model save at {trained_model_path}")
    return trained_model_path


@app.route('/predict', methods=['POST'])
def predict_route() -> Response:
    data = request.json
    predict_request = PredictRequest(**data)
    df = load_data(predict_request.input_path)
    preprocessed_data = preprocess_data(df)
    predictions = predict(preprocessed_data, predict_request.model_path)
    response = {'predictions': predictions.tolist()}
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
