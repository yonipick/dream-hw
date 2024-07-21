import joblib


def load_model(model_path):
    model = joblib.load(model_path)
    return model


def predict(input_data, model_path: str):
    model = load_model(model_path)
    predictions = model.predict(input_data)
    return predictions
