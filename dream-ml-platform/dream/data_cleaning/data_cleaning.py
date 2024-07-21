import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def create_encoder(dataset: pd.DataFrame) -> None:
    OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    object_cols = dataset.select_dtypes(include='object').columns
    OH_encoder.fit(dataset[object_cols])
    encoder_path = os.path.join(Path.cwd().parent, "models", "encoder.pkl")
    joblib.dump(OH_encoder, encoder_path)
    train_data = dataset.loc[lambda dx: dx.SalePrice.notna()]
    test_data = dataset.loc[lambda dx: dx.SalePrice.isna()].drop(['SalePrice'], axis=1)
    train_data.to_csv(os.path.join(Path.cwd().parent, "dataframes", 'train_data.csv'), index=True)
    test_data.to_csv(os.path.join(Path.cwd().parent, "dataframes", 'test_data.csv'), index=True)


if __name__ == '__main__':
    dataset_url = "https://drive.google.com/file/d/1kqnB4J8FuF1k8xLIvfbPE1jsqUwd3wVH/view?usp=sharing"
    path = f"https://drive.google.com/uc?export=download&id=" \
           f"{dataset_url.split('/')[-2]}"
    df = pd.read_csv(path).set_index('Id')
    cleaning(df)
