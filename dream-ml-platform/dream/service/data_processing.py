import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def load_data(path: str) -> pd.DataFrame:
    dataset = pd.read_csv(path).set_index('Id')
    return dataset


def create_encoder(dataset: pd.DataFrame) -> None:
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    object_cols = dataset.select_dtypes(include='object').columns
    one_hot_encoder.fit(dataset[object_cols])
    encoder_path = os.path.join(Path.cwd().parent, "models", "encoder.pkl")
    joblib.dump(one_hot_encoder, encoder_path)
    train_data = dataset.loc[lambda dx: dx.SalePrice.notna()]
    test_data = dataset.loc[lambda dx: dx.SalePrice.isna()].drop(['SalePrice'], axis=1)
    train_data.to_csv(os.path.join(Path.cwd().parent, "dataframes", 'train_data.csv'), index=True)
    test_data.to_csv(os.path.join(Path.cwd().parent, "dataframes", 'test_data.csv'), index=True)


def preprocess_data(dataset: pd.DataFrame) -> pd.DataFrame:
    encoder_path = os.path.join(os.getcwd(), "encoder.pkl")
    one_hot_encoder = joblib.load(encoder_path)
    object_cols = dataset.select_dtypes(include='object').columns
    one_hot_cols = pd.DataFrame(one_hot_encoder.transform(dataset[object_cols]))
    one_hot_cols.index = dataset.index
    one_hot_cols.columns = one_hot_encoder.get_feature_names_out()
    df_final = dataset.drop(object_cols, axis=1)
    df_final = pd.concat([df_final, one_hot_cols], axis=1)
    return df_final
