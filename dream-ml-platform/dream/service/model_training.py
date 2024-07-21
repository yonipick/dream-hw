import os
from pathlib import Path

import numpy as np
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib


def score_func(ytrue, ypred):
    return np.sqrt(metrics.mean_squared_error(ytrue, ypred))


def train_model(df, model_type='RFR'):
    X = df.loc[lambda dx: dx.SalePrice.notna()].drop(['SalePrice'], axis=1)
    Y = df.loc[lambda dx: dx.SalePrice.notna()].loc[:, 'SalePrice']
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2,
                                                          random_state=0)

    if model_type == 'RFR':
        model = RandomForestRegressor(n_estimators=10)
    elif model_type == 'LR':
        model = LinearRegression()
    elif model_type == 'SVR':
        model = svm.SVR()
    else:
        raise ValueError("Model type not supported")

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_valid)
    print(f"{model_type} Score:", score_func(Y_valid, Y_pred))
    model_path = os.path.join(Path.cwd().parent, "models", f"{model_type}_model.pkl")
    joblib.dump(model, model_path)
    return model_path
