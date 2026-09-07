"""Separate response-level HaluEval TF-IDF + Logistic Regression baseline."""
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.evaluation.evaluate_classifier import evaluate_predictions


def load_halueval(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def run_halueval_baseline(path, seed=42, validation_size=.2):
    records = load_halueval(path)
    labels = [record["label"] for record in records]
    train, valid = train_test_split(records, test_size=validation_size, random_state=seed, stratify=labels)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True, min_df=2)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    model.fit(vectorizer.fit_transform([record["text"] for record in train]), [record["label"] for record in train])
    features = vectorizer.transform([record["text"] for record in valid])
    predicted = model.predict(features)
    details = [{"confidence": float(max(row)), "nli_label": None} for row in model.predict_proba(features)]
    return evaluate_predictions([record["label"] for record in valid], list(predicted), "HaluEval_TFIDF_LogReg",
                                valid, prediction_details=details)
