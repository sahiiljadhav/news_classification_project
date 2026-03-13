# News Article Classification

## Project Overview
This project classifies news articles into predefined categories using a machine learning pipeline written entirely in Python scripts. It follows the full workflow of preprocessing, TF-IDF feature extraction, Logistic Regression training, and evaluation from the terminal.

## Dataset Source
- Dataset: BBC News Dataset
- Source: https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook/blob/master/Chapter04/bbc-text.csv
- Categories: business, entertainment, politics, sport, tech

## Folder Structure Explanation
```text
news_classification_project/
│
├── data/
│   ├── raw/
│   │   └── bbc-text.csv
│   └── processed/
│       ├── feature_engineered.pkl
│       ├── preprocessed_data.pkl
│       └── tfidf_vectorizer.pkl
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── config.py
│
├── models/
│   └── news_classifier.pkl
│
├── results/
│   └── metrics.txt
│
├── requirements.txt
├── README.md
└── main.py
```

## Steps to Run the Project
1. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
2. Place the dataset file at data/raw/bbc-text.csv.
3. Run the full pipeline from the project root:
   ```bash
   python main.py
   ```

## Model Used
- Algorithm: Logistic Regression
- Feature Extraction: TF-IDF with 5000 features

## Final Result Summary
The current verified run achieves an accuracy of 0.9843 on the held-out test set. The confusion matrix and accuracy are saved in results/metrics.txt.
