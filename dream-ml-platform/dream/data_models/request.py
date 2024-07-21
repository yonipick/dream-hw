from typing import Any

from pydantic import Field

from dream.data_models.base import SharedBaseModel


class TrainRequest(SharedBaseModel):
    dataset_path: str
    model_type: str = Field(
        description="Name of the model to train"
                    " [RFR for Random Forest Regressor, LR for LinearRegression, SVR]",
        default="RFR",
    )


class PredictRequest(SharedBaseModel):
    input_path: str
    model_path: str
