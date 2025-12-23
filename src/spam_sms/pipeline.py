"""Training, evaluation, and prediction pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import joblib

from . import data

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
VECTORIZER_PATH = ARTIFACTS_DIR / "vectorizer.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


def load_dataset(test_size: float = 0.2, seed: int = 7) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    df = data.load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=test_size, stratify=df["label"], random_state=seed
    )
    return X_train, X_test, y_train, y_test


def build_pipeline() -> Pipeline:
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("vectorizer", vectorizer), ("clf", model)])


def train(test_size: float = 0.2, seed: int = 7) -> Dict[str, float]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_dataset(test_size=test_size, seed=seed)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    metrics = {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }

    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


def predict(message: str, threshold: float = 0.5) -> Dict[str, float | str]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not trained. Run train first.")
    pipeline: Pipeline = joblib.load(MODEL_PATH)
    proba = pipeline.predict_proba([message])[0, 1]
    label = "spam" if proba >= threshold else "ham"
    return {"probability": float(proba), "label": label}
