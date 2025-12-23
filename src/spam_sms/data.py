"""Dataset download and loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TRAIN_URL = "https://cdn.freecodecamp.org/project-data/sms/train-data.tsv"
VALID_URL = "https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv"
TRAIN_PATH = DATA_DIR / "train-data.tsv"
VALID_PATH = DATA_DIR / "valid-data.tsv"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {dest.name}") as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def ensure_data() -> Tuple[Path, Path]:
    _download(TRAIN_URL, TRAIN_PATH)
    _download(VALID_URL, VALID_PATH)
    return TRAIN_PATH, VALID_PATH


def load_data() -> pd.DataFrame:
    train_path, valid_path = ensure_data()
    train_df = pd.read_csv(train_path, sep="\t", header=None, names=["label", "message"])
    valid_df = pd.read_csv(valid_path, sep="\t", header=None, names=["label", "message"])
    df = pd.concat([train_df, valid_df], ignore_index=True)
    df = df[df["message"].str.strip() != ""].copy()
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df
