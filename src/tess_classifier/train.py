from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    DATA_DIR,
    LOCAL_TESS_FEATURE_DATA_FILE,
    METRICS_FILE,
    MODEL_FILE,
    MODELS_DIR,
    SYNTHETIC_FEATURE_DATA_FILE,
    TESS_CURATED_DIR,
)
from .data import build_demo_dataset, build_local_tess_dataset
from .data import summarize_local_dataset
from .features import FEATURE_COLUMNS


def _build_training_dataset(
    dataset_source: str,
    samples_per_class: int,
    seed: int,
    dataset_dir: Path,
) -> tuple:
    if dataset_source == "synthetic":
        dataset = build_demo_dataset(samples_per_class=samples_per_class, seed=seed)
        feature_file = SYNTHETIC_FEATURE_DATA_FILE
    elif dataset_source == "local_tess":
        dataset = build_local_tess_dataset(dataset_dir=dataset_dir)
        feature_file = LOCAL_TESS_FEATURE_DATA_FILE
    else:
        raise ValueError(f"Unsupported dataset source: {dataset_source}")

    return dataset, feature_file


def _validate_dataset(dataset) -> None:
    label_counts = dataset["label"].value_counts()
    if label_counts.shape[0] < 2:
        raise ValueError("Training requires at least two distinct labels.")
    if (label_counts < 2).any():
        too_small = ", ".join(label_counts[label_counts < 2].index.tolist())
        raise ValueError(f"Each label needs at least 2 samples. Too small: {too_small}")
    test_rows = max(math.ceil(len(dataset) * 0.2), int(label_counts.shape[0]))
    train_rows = len(dataset) - test_rows
    if train_rows < label_counts.shape[0]:
        raise ValueError(
            "Dataset is too small for a stratified split with at least one training example per class. "
            "Add more samples so both train and test sets can include every class."
        )


def _choose_test_size(dataset) -> int:
    """Pick a stratified test size that works for small curated datasets."""
    num_labels = int(dataset["label"].nunique())
    return max(math.ceil(len(dataset) * 0.2), num_labels)


def train_and_select_model(
    dataset_source: str,
    samples_per_class: int,
    seed: int,
    dataset_dir: Path,
) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    dataset, feature_file = _build_training_dataset(
        dataset_source=dataset_source,
        samples_per_class=samples_per_class,
        seed=seed,
        dataset_dir=dataset_dir,
    )
    _validate_dataset(dataset)
    dataset.to_csv(feature_file, index=False)

    X = dataset[list(FEATURE_COLUMNS)]
    y = dataset["label"]
    test_size = _choose_test_size(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    candidates = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=seed)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            random_state=seed,
        ),
    }

    results: dict[str, dict] = {}
    best_name = ""
    best_model = None
    best_f1 = -1.0

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(y_test, predictions, average="macro")
        report = classification_report(y_test, predictions, output_dict=True)

        results[name] = {
            "accuracy": round(float(accuracy), 4),
            "macro_f1": round(float(macro_f1), 4),
            "report": report,
        }

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_name = name
            best_model = model

    if best_model is None:
        raise RuntimeError("Model training failed to produce a best model.")

    artifact = {
        "model": best_model,
        "best_model_name": best_name,
        "feature_columns": list(FEATURE_COLUMNS),
        "metrics": results,
    }
    joblib.dump(artifact, MODEL_FILE)

    metrics_payload = {
        "dataset_source": dataset_source,
        "dataset_rows": int(len(dataset)),
        "feature_file": str(feature_file),
        "dataset_dir": str(dataset_dir),
        "samples_per_class": samples_per_class,
        "test_rows": test_size,
        "best_model_name": best_name,
        "results": results,
    }
    if dataset_source == "local_tess":
        metrics_payload["dataset_summary"] = summarize_local_dataset(dataset_dir)
    METRICS_FILE.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline stellar-variability classifier.")
    parser.add_argument("--dataset-source", choices=("synthetic", "local_tess"), default="synthetic")
    parser.add_argument("--dataset-dir", type=Path, default=TESS_CURATED_DIR)
    parser.add_argument("--samples-per-class", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics = train_and_select_model(
        dataset_source=args.dataset_source,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        dataset_dir=args.dataset_dir,
    )
    print("Training complete.")
    print(f"Dataset source: {metrics['dataset_source']}")
    print(f"Best model: {metrics['best_model_name']}")
    for model_name, values in metrics["results"].items():
        print(
            f"{model_name}: accuracy={values['accuracy']:.4f}, "
            f"macro_f1={values['macro_f1']:.4f}"
        )
    print(f"Saved model artifact to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
