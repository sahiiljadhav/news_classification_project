import joblib
from sklearn.metrics import accuracy_score, confusion_matrix

from config import FEATURE_DATA_PATH, METRICS_PATH, MODEL_PATH, RESULTS_DIR


def evaluate():
    model = joblib.load(MODEL_PATH)
    data = joblib.load(FEATURE_DATA_PATH)

    y_pred = model.predict(data["X_test_tfidf"])
    accuracy = accuracy_score(data["y_test"], y_pred)
    matrix = confusion_matrix(data["y_test"], y_pred)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        metrics_file.write(f"Accuracy: {accuracy:.4f}\n")
        metrics_file.write("Confusion Matrix:\n")
        metrics_file.write(str(matrix))

    print(f"Evaluation completed. Accuracy = {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    evaluate()