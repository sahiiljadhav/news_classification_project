import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from config import FEATURE_DATA_PATH, PREPROCESSED_PATH, TFIDF_MAX_FEATURES, VECTORIZER_PATH


def engineer_features():
    data = joblib.load(PREPROCESSED_PATH)
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)

    data["X_train_tfidf"] = vectorizer.fit_transform(data["X_train"])
    data["X_test_tfidf"] = vectorizer.transform(data["X_test"])

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(data, FEATURE_DATA_PATH)
    print("Feature engineering completed.")


if __name__ == "__main__":
    engineer_features()