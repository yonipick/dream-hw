from dream.data_models.base import SharedBaseModel


class PredictRequest(SharedBaseModel):
    input_path: str
