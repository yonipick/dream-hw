from dream.data_models.base import SharedBaseModel


class TrainResponse(SharedBaseModel):
    model_path: str
    model_score: float
