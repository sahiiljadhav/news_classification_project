import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split

from config import DATASET_PATH, PREPROCESSED_PATH, PROCESSED_DATA_DIR, RANDOM_STATE, RAW_DATA_DIR


STOP_WORDS = set(ENGLISH_STOP_WORDS)


def ensure_directories():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text):
    normalized = str(text).lower()
    normalized = re.sub(r"[^a-zA-Z\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return " ".join(word for word in normalized.split() if word not in STOP_WORDS)


def preprocess():
    ensure_directories()
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Place the CSV there before running the pipeline."
        )

    dataset = pd.read_csv(DATASET_PATH)
    dataset.columns = [column.strip().lower() for column in dataset.columns]

    required_columns = {"category", "text"}
    if not required_columns.issubset(dataset.columns):
        missing_columns = sorted(required_columns.difference(dataset.columns))
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(missing_columns)}"
        )

    dataset = dataset.dropna(subset=["category", "text"]).copy()
    dataset["text"] = dataset["text"].map(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset["text"],
        dataset["category"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=dataset["category"],
    )

    joblib.dump(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        },
        PREPROCESSED_PATH,
    )
    print(f"Preprocessing completed. Saved to {PREPROCESSED_PATH}.")


if __name__ == "__main__":
    preprocess()