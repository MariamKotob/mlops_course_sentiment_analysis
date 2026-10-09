"""BERT encoder with a small feed-forward classification head."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from mlops_practitioner_course.config import ModelConfig


class BertClassifier(nn.Module):  # pragma: no cover
    """Classifies a text from the final hidden state of its [CLS] token."""

    def __init__(
        self,
        model_name: str,
        hidden_dim: int = 50,
        num_labels: int = 2,
        dropout: float = 0.5,
        freeze_bert: bool = False,
        pretrained: bool = True,
        local_files_only: bool = False,
    ) -> None:
        """
        Args:
            model_name: Hugging Face model id of the BERT encoder.
            hidden_dim: Size of the classifier head's hidden layer.
            num_labels: Number of output classes.
            dropout: Dropout probability inside the classifier head.
            freeze_bert: If True, only the classifier head is trained.
            pretrained: Load pretrained encoder weights. Set False when the weights
                will be overwritten by a checkpoint, to skip the extra download/load.
        """
        super().__init__()
        if pretrained:
            self.bert = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        else:
            self.bert = AutoModel.from_config(AutoConfig.from_pretrained(model_name, local_files_only=local_files_only))

        # Read the encoder width from its config instead of hard-coding 256 / 768.
        bert_dim = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    @classmethod
    def from_config(cls, config: ModelConfig, pretrained: bool = True, local_files_only: bool = False, model_path: str | None = None) -> BertClassifier:
        return cls(
            model_name=model_path or config.name,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            freeze_bert=config.freeze_bert,
            pretrained=pretrained,
            local_files_only=local_files_only,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Map (batch, seq_len) token ids and mask to (batch, num_labels) logits."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden_state = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_hidden_state)
