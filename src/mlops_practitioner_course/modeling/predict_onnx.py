from __future__ import annotations

import numpy as np
import onnxruntime as ort
import yaml
from pathlib import Path
from collections.abc import Sequence

from mlops_practitioner_course.preprocess import BertPreprocessor

class OnnxSentimentPredictor:
    """Inference wrapper for the ONNX format model."""
    LABEL_NAMES = ("negative", "positive")

    def __init__(
        self,
        session: ort.InferenceSession,
        preprocessor: BertPreprocessor,
        threshold: float = 0.5,
    ) -> None:
        self.session = session
        self.preprocessor = preprocessor
        self.threshold = threshold

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> "OnnxSentimentPredictor":
        """Load the ONNX model and the preprocessing settings."""
        path = Path(path)
        checkpoint_dir = path.parent
        
        # Configure ONNX Runtime to use CPU or GPU
        providers = ['CPUExecutionProvider']
        if "cuda" in device:
            providers = ['CUDAExecutionProvider'] + providers
            
        session = ort.InferenceSession(str(checkpoint_dir / "model.onnx"), providers=providers)
        
        # Load settings from settings.yaml to rebuild the preprocessor without PyTorch
        from mlops_practitioner_course.config import Settings
        
        settings_path = checkpoint_dir / "settings.yaml"
        with open(settings_path) as f:
            settings_dict = yaml.safe_load(f)
        settings = Settings.model_validate(settings_dict)
        
        return cls(
            session=session,
            preprocessor=BertPreprocessor.from_settings(settings, local_files_only=True, model_path=str(checkpoint_dir)),
            threshold=settings.evaluation.threshold,
        )

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        # Encode with HuggingFace tokenizer (returns numpy arrays now)
        input_ids, attention_mask = self.preprocessor.encode(texts)
        
        # Run inference using ONNX Runtime
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        logits = self.session.run(["logits"], inputs)[0]
        
        # Convert logits to probabilities (Softmax)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs[:, 1]
