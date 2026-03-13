import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import data_preprocessing
import evaluate
import feature_engineering
import train


def main():
    print("Starting pipeline...")
    data_preprocessing.preprocess()
    feature_engineering.engineer_features()
    train.train_model()
    accuracy = evaluate.evaluate()
    print(f"Final accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()