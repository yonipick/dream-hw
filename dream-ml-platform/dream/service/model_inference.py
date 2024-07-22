import os.path
from pathlib import Path
from typing import Any, List

import joblib
import pandas as pd


class ModelInference:
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)

    @staticmethod
    def load_model(model_path: str) -> Any:
        model = joblib.load(model_path)
        return model

    def inference(self, input_data: pd.DataFrame) -> List[float]:
        predictions = self.model.predict(input_data)
        return predictions.tolist()
