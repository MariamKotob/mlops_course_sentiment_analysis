"""Command-line entry point: train, evaluate or predict with the sentiment model.

    uv run mlops-practitioner-course train
    uv run mlops-practitioner-course evaluate
    uv run mlops-practitioner-course predict "تغريدة" "تغريدة أخرى"
"""

import argparse
import json
import logging
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader

if not torch.cuda.is_available():
    torch.cuda.is_current_stream_capturing = lambda: False

from mlops_practitioner_course.config import DEFAULT_CONFIG_PATH, Settings
from mlops_practitioner_course.data import ArabicTweetsLoader
from mlops_practitioner_course.modeling.checkpoint import CHECKPOINT_FILENAME, save_checkpoint
from mlops_practitioner_course.modeling.evaluate import EvaluationReport, evaluate_predictions, plot_roc
from mlops_practitioner_course.modeling.model import BertClassifier
from mlops_practitioner_course.modeling.predict import SentimentPredictor
from mlops_practitioner_course.modeling.train import Trainer, resolve_device, set_seed
from mlops_practitioner_course.preprocess import BertPreprocessor

logger = logging.getLogger(__name__)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    # Hugging Face Hub logs every HTTP request at INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def evaluate_split(
    predictor: SentimentPredictor,
    loader: DataLoader,
    labels: list[int],
    split_name: str,
    out_dir: Path,
) -> EvaluationReport:
    """Predict a split, log its metrics and save metrics JSON + ROC plot to `out_dir`."""
    probs = predictor.predict_proba_loader(loader)
    report = evaluate_predictions(labels, probs, predictor.threshold)
    logger.info(
        "%s: accuracy %.2f%% | AUC %.4f | F1 %.4f | n=%d",
        split_name, 100 * report.accuracy, report.roc_auc, report.f1, report.n_samples,
    )
    report.save(out_dir / f"metrics_{split_name}.json")
    plot_roc(labels, probs, f"ROC: {out_dir.name} on {split_name}", out_dir / f"roc_{split_name}.png")
    return report


def train(settings: Settings) -> None:
    set_seed(settings.seed)
    device = resolve_device(settings.training.device)
    out_dir = settings.run_dir

    loader = ArabicTweetsLoader.from_settings(settings)
    train_split, val_split = loader.load_train_val()
    test_split = loader.load_test()

    preprocessor = BertPreprocessor.from_settings(settings)
    train_loader = preprocessor.build_dataloader(train_split.texts, train_split.labels, shuffle=True)
    val_loader = preprocessor.build_dataloader(val_split.texts, val_split.labels)
    test_loader = preprocessor.build_dataloader(test_split.texts, test_split.labels)

    model = BertClassifier.from_config(settings.model)
    trainer = Trainer(model, settings.training, device, settings=settings)
    history = trainer.fit(train_loader, val_loader)

    save_checkpoint(model, preprocessor, settings, out_dir / CHECKPOINT_FILENAME)
    (out_dir / "history.json").write_text(
        json.dumps([epoch.to_dict() for epoch in history], indent=2), encoding="utf-8"
    )

    predictor = SentimentPredictor(model, preprocessor, device, settings.evaluation.threshold)
    evaluate_split(predictor, val_loader, val_split.labels, "val", out_dir)
    evaluate_split(predictor, test_loader, test_split.labels, "test", out_dir)

    if getattr(trainer, "run_id", None):
        with mlflow.start_run(run_id=trainer.run_id):
            for artifact_name in [
                "metrics_val.json",
                "roc_val.png",
                "metrics_test.json",
                "roc_test.png",
                CHECKPOINT_FILENAME,
                "history.json",
            ]:
                artifact_path = out_dir / artifact_name
                if artifact_path.exists():
                    mlflow.log_artifact(str(artifact_path))


def evaluate(settings: Settings, checkpoint: Path) -> None:
    """Re-score a saved model on the test split, without retraining."""
    predictor = SentimentPredictor.from_checkpoint(checkpoint, settings.training.device)
    test_split = ArabicTweetsLoader.from_settings(settings).load_test()
    test_loader = predictor.preprocessor.build_dataloader(test_split.texts, test_split.labels)
    evaluate_split(predictor, test_loader, test_split.labels, "test", checkpoint.parent)


def predict(settings: Settings, checkpoint: Path, texts: list[str]) -> None:
    predictor = SentimentPredictor.from_checkpoint(checkpoint, settings.training.device)
    probs = predictor.predict_proba(texts)
    for text, prob in zip(texts, probs):
        label = predictor.LABEL_NAMES[int(prob >= predictor.threshold)]
        # Results are the command's output, so they go to stdout rather than the log.
        print(f"{label}\t{prob:.3f}\t{text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Arabic tweet sentiment pipeline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the YAML config")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Train, save and evaluate a model")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a saved model on the test split")
    eval_parser.add_argument("--checkpoint", type=Path, help="Defaults to the configured run's model.pt")

    predict_parser = subparsers.add_parser("predict", help="Predict the sentiment of texts")
    predict_parser.add_argument("texts", nargs="+")
    predict_parser.add_argument("--checkpoint", type=Path, help="Defaults to the configured run's model.pt")

    args = parser.parse_args()
    settings = Settings.from_yaml(args.config)
    setup_logging(settings.logging.level)
    logger.info("Loaded config from %s", args.config)

    checkpoint = getattr(args, "checkpoint", None) or settings.run_dir / CHECKPOINT_FILENAME
    if args.command == "train":
        train(settings)
    elif args.command == "evaluate":
        evaluate(settings, checkpoint)
    elif args.command == "predict":
        predict(settings, checkpoint, args.texts)


if __name__ == "__main__":
    # App entry point
    main()
