from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

DATASET_PATH = RAW_DATA_DIR / "bbc-text.csv"
PREPROCESSED_PATH = PROCESSED_DATA_DIR / "preprocessed_data.pkl"
FEATURE_DATA_PATH = PROCESSED_DATA_DIR / "feature_engineered.pkl"
VECTORIZER_PATH = PROCESSED_DATA_DIR / "tfidf_vectorizer.pkl"
MODEL_PATH = MODELS_DIR / "news_classifier.pkl"
METRICS_PATH = RESULTS_DIR / "metrics.txt"

DATASET_SOURCE = "https://storage.googleapis.com/dataset-uploader/bbc/bbc-text.csv"
RANDOM_STATE = 42
TFIDF_MAX_FEATURES = 5000