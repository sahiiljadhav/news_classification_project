import joblib
from sklearn.linear_model import LogisticRegression

from config import FEATURE_DATA_PATH, MODEL_PATH, MODELS_DIR, RANDOM_STATE


def train_model():
    data = joblib.load(FEATURE_DATA_PATH)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(data["X_train_tfidf"], data["y_train"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Model training completed and saved.")


if __name__ == "__main__":
    train_model()