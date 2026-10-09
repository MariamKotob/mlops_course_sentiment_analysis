"""Saving and loading trained models.

A checkpoint stores the model's state_dict together with the Settings it was trained
with, so the exact model architecture and preprocessing can be rebuilt at load time.
This replaces the notebook's pickle.dump(model), which breaks when code moves and can
execute arbitrary code on load.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from mlops_practitioner_course.config import Settings
from mlops_practitioner_course.modeling.model import BertClassifier

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = "model.pt"


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mlops_practitioner_course.preprocess import BertPreprocessor

def save_checkpoint(model: BertClassifier, preprocessor: "BertPreprocessor", settings: Settings, path: str | Path) -> Path:  # pragma: no cover
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the tokenizer and config into the checkpoint directory so it's fully self-contained
    model.bert.config.save_pretrained(path.parent)
    preprocessor.tokenizer.save_pretrained(path.parent)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "settings": settings.model_dump(mode="json"),
        },
        path,
    )
    logger.info("Saved checkpoint to %s", path)
    return path


def load_checkpoint(  # pragma: no cover
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[BertClassifier, Settings]:
    """Rebuild the model from a checkpoint and return it with its training Settings."""
    # weights_only=True refuses to unpickle arbitrary objects.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    settings = Settings.model_validate(checkpoint["settings"])

    # pretrained=False: the encoder weights come from the checkpoint, not the Hub.
    model = BertClassifier.from_config(
        settings.model,
        pretrained=False,
        local_files_only=True,
        model_path=str(Path(path).parent),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    logger.info("Loaded checkpoint from %s (%s)", path, settings.model.name)
    return model, settings
