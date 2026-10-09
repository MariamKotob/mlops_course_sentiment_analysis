"""Inference with a trained BertClassifier."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mlops_practitioner_course.modeling.checkpoint import load_checkpoint
from mlops_practitioner_course.modeling.model import BertClassifier
from mlops_practitioner_course.modeling.train import resolve_device
from mlops_practitioner_course.preprocess import BertPreprocessor


class SentimentPredictor:  # pragma: no cover
    """Turns raw tweets into positive-class probabilities and sentiment labels."""

    LABEL_NAMES = ("negative", "positive")

    def __init__(
        self,
        model: BertClassifier,
        preprocessor: BertPreprocessor,
        device: torch.device,
        threshold: float = 0.5,
    ) -> None:
        self.model = model.to(device).eval()
        self.preprocessor = preprocessor
        self.device = device
        self.threshold = threshold

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto") -> SentimentPredictor:
        """Load a model plus the same preprocessing it was trained with."""
        torch_device = resolve_device(device)
        model, settings = load_checkpoint(path, torch_device)
        return cls(
            model=model,
            preprocessor=BertPreprocessor.from_settings(
                settings,
                local_files_only=True,
                model_path=str(Path(path).parent),
            ),
            device=torch_device,
            threshold=settings.evaluation.threshold,
        )

    @torch.no_grad()
    def predict_proba_loader(self, loader: DataLoader) -> np.ndarray:
        """P(positive) for every sample in `loader`, in loader order (do not shuffle)."""
        self.model.eval()
        probs = []
        for batch in loader:
            input_ids, attention_mask = (t.to(self.device) for t in batch[:2])
            logits = self.model(input_ids, attention_mask)
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
        return torch.cat(probs).numpy()

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self.predict_proba_loader(self.preprocessor.build_dataloader(texts))

    def predict(self, texts: Sequence[str]) -> list[str]:
        """Sentiment label ("negative" / "positive") for each text."""
        positive = self.predict_proba(texts) >= self.threshold
        return [self.LABEL_NAMES[int(p)] for p in positive]


