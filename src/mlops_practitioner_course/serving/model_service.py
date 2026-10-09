import bentoml
import numpy as np
from mlops_practitioner_course.config import Settings
from mlops_practitioner_course.modeling.predict_onnx import OnnxSentimentPredictor

@bentoml.service(resources={"cpu": "2"}, traffic={"timeout": 60})
class ModelService:
    def __init__(self) -> None:
        self.settings = Settings.from_yaml()
        self.model = OnnxSentimentPredictor.from_checkpoint(
            self.settings.run_dir / "model.pt", self.settings.training.device
        )

    @bentoml.api(batchable=True, max_batch_size=64, max_latency_ms=50)
    def predict(self, texts: list[str]) -> list[float]:
        # predict_proba expects a Sequence[str] and returns a NumPy array
        probs = self.model.predict_proba(texts)
        return probs.tolist()

    @bentoml.api
    def get_threshold(self) -> float:
        return self.model.threshold
