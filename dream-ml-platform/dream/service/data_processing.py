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
    encoder_path = os.path.join(Path.cwd(), "models")
    if not os.path.isdir(Path(encoder_path)):
        os.makedirs(Path(encoder_path), exist_ok=True)
    try:
        joblib.dump(one_hot_encoder, os.path.join(encoder_path, "encoder.pkl"))
        print(f"Encoder saved to {encoder_path}")
    except Exception as error:
        print(f"Failed to save file {encoder_path}: {error}")

    train_data = dataset.loc[lambda dx: dx.SalePrice.notna()]
    test_data = dataset.loc[lambda dx: dx.SalePrice.isna()].drop(['SalePrice'], axis=1)
    dataframe_path = os.path.join(Path.cwd(), "dataframes")
    if not os.path.isdir(Path(dataframe_path)):
        os.makedirs(Path(dataframe_path), exist_ok=True)
    try:
        train_data.to_csv(os.path.join(dataframe_path, 'train_data.csv'), index=True)
        test_data.to_csv(os.path.join(dataframe_path, 'test_data.csv'), index=True)
        print(f"Saved datasets in {dataframe_path}")
    except Exception as error:
        print(f"Failed to save files {dataframe_path}: {error}")


def preprocess_data(dataset: pd.DataFrame) -> pd.DataFrame:
    encoder_path = os.path.join(Path(os.getcwd()), "models", "encoder.pkl")
    one_hot_encoder = joblib.load(encoder_path)
    object_cols = dataset.select_dtypes(include='object').columns
    one_hot_cols = pd.DataFrame(one_hot_encoder.transform(dataset[object_cols]))
    one_hot_cols.index = dataset.index
    one_hot_cols.columns = one_hot_encoder.get_feature_names_out()
    df_final = dataset.drop(object_cols, axis=1)
    df_final = pd.concat([df_final, one_hot_cols], axis=1)
    return df_final


if __name__ == '__main__':
    dataset_url = "https://drive.google.com/file/d/1kqnB4J8FuF1k8xLIvfbPE1jsqUwd3wVH/view?usp=sharing"
    path = f"https://drive.google.com/uc?export=download&id=" \
           f"{dataset_url.split('/')[-2]}"
    df = pd.read_csv(path).set_index('Id')
    create_encoder(df)
