import os
from pathlib import Path

from flask import Flask, request, jsonify, Response
from flask_injector import FlaskInjector
from injector import inject, singleton, Binder

from dream.data_models.request import PredictRequest
from dream.service.data_processing import preprocess_data, load_data
from dream.service.model_inference import ModelInference

app = Flask(__name__)


# FlaskInjector setup
def configure(binder: Binder):
    model_path = './models/model.pkl'
    binder.bind(
        ModelInference,
        to=ModelInference(model_path),
        scope=singleton
    )


@app.route("/")
def home():
    return "<h1>Dream home assessment</h1>"


@app.route('/predict', methods=['POST'])
@inject
def predict_route(model_inference: ModelInference) -> Response:
    predict_request = PredictRequest(**request.json)
    df = load_data(predict_request.input_path)
    preprocessed_data = preprocess_data(df)
    predictions = model_inference.inference(preprocessed_data)
    df["SalePrice"] = predictions
    result_path = os.path.join(Path.cwd(), "dataframes", 'result_data.csv')
    df.to_csv(result_path, index=True)
    response = {'result_path': result_path}
    return jsonify(response)


FlaskInjector(app=app, modules=[configure])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
