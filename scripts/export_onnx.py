"""Export a trained PyTorch model to ONNX format for faster inference."""

import argparse
from pathlib import Path
import torch

from mlops_practitioner_course.modeling.predict import SentimentPredictor


def export_to_onnx(checkpoint_dir: Path, output_file: str = "model.onnx") -> None:
    # 1. Load the PyTorch model and settings
    print(f"Loading PyTorch model from {checkpoint_dir}...")
    predictor = SentimentPredictor.from_checkpoint(checkpoint_dir / "model.pt", device="cpu")
    model = predictor.model
    model.eval()

    # 2. Create dummy inputs matching the tokenizer's max_length
    max_length = predictor.preprocessor.max_length
    dummy_input_ids = torch.zeros((1, max_length), dtype=torch.long, device="cpu")
    dummy_attention_mask = torch.ones((1, max_length), dtype=torch.long, device="cpu")

    # 3. Export to ONNX
    onnx_path = checkpoint_dir / output_file
    print(f"Exporting to ONNX format at {onnx_path}...")
    
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
    )
    
    print("ONNX export complete!")
    print(f"The ONNX model is saved alongside the tokenizer files in: {checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/bert-mini"), 
                        help="Directory containing model.pt and tokenizer files")
    args = parser.parse_args()
    
    export_to_onnx(args.checkpoint_dir)
