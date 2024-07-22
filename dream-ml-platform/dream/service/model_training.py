import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

from dream.service.data_processing import create_encoder, preprocess_data


def score_func(ytrue: Any, ypred: Any) -> Any:
    return np.sqrt(metrics.mean_squared_error(ytrue, ypred))


def train_model(dataset: pd.DataFrame) -> None:
    X = dataset.loc[lambda dx: dx.SalePrice.notna()].drop(['SalePrice'], axis=1)
    Y = dataset.loc[lambda dx: dx.SalePrice.notna()].loc[:, 'SalePrice']
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2,
                                                          random_state=0)
    models = {
        "RFR": RandomForestRegressor(n_estimators=10),
        "LR": LinearRegression(),
        "SVR": svm.SVR()
    }
    models_result = {}
    for model_type, model in models.items():
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_valid)
        model_score = score_func(Y_valid, y_pred)
        print(f"{model_type} Score:", model_score)
        models_result[model_type] = model_score

    best_model_type = min(models_result, key=models_result.get)
    best_model = models[best_model_type]
    best_model.fit(X_train, Y_train)

    # Save the best model
    model_path = os.path.join(Path.cwd(), "models")
    if not os.path.isdir(Path(model_path)):
        os.makedirs(Path(model_path), exist_ok=True)
    try:
        joblib.dump(best_model, os.path.join(model_path, "model.pkl"))
    except Exception as error:
        print(f"Failed to save model {model_path}: {error}")
    print(f"Save {best_model_type} model in {model_path}")


if __name__ == '__main__':
    dataset_url = "https://drive.google.com/file/d/1kqnB4J8FuF1k8xLIvfbPE1jsqUwd3wVH/view?usp=sharing"
    path = f"https://drive.google.com/uc?export=download&id=" \
           f"{dataset_url.split('/')[-2]}"
    df = pd.read_csv(path).set_index('Id')
    create_encoder(df)
    preprocessed_data = preprocess_data(df)
    train_model(preprocessed_data)
