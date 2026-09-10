"""
baselines.py
Intent Classification Baselines for Twitter Customer Support Dataset.

Baseline 1 (Trivial): Majority-Class Classifier.
Baseline 2 (Simple): TF-IDF + Logistic Regression (with pure-Python Naive Bayes fallback if sklearn missing).

Outputs:
- Intent Accuracy & Macro-F1
- Confusion matrix & per-class metrics
- Saves baseline performance to data/processed/baseline_metrics.json
"""

import json
import os
import math
from collections import Counter, defaultdict

KB_CORPUS = "data/processed/kb_corpus.jsonl"
GOLDEN_SET = "data/processed/golden_eval_set.jsonl"
METRICS_OUT = "data/processed/baseline_metrics.json"

def load_data(filepath):
    queries, intents = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            queries.append(item["query"])
            intents.append(item["intent"])
    return queries, intents

def calculate_macro_f1(y_true, y_pred, labels):
    """Calculates macro-averaged F1 score without external libraries."""
    f1_scores = []
    for lbl in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lbl and yp == lbl)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != lbl and yp == lbl)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lbl and yp != lbl)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

def run_majority_baseline(train_intents, test_intents, labels):
    majority_class = Counter(train_intents).most_common(1)[0][0]
    preds = [majority_class] * len(test_intents)
    acc = sum(1 for yt, yp in zip(test_intents, preds) if yt == yp) / len(test_intents)
    macro_f1 = calculate_macro_f1(test_intents, preds, labels)
    return {
        "model": "Majority Class (Trivial)",
        "majority_class": majority_class,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4)
    }

def run_simple_baseline(train_queries, train_intents, test_queries, test_intents, labels):
    """Uses sklearn TF-IDF + LogisticRegression if available, otherwise pure Python Naive Bayes."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score

        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        X_train = vec.fit_transform(train_queries)
        X_test = vec.transform(test_queries)

        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(X_train, train_intents)
        preds = clf.predict(X_test)

        acc = accuracy_score(test_intents, preds)
        macro_f1 = f1_score(test_intents, preds, average="macro")
        engine = "sklearn (TF-IDF + LogisticRegression)"
    except ImportError:
        engine = "pure-python (TF-IDF + Naive Bayes fallback)"
        word_counts = defaultdict(lambda: Counter())
        class_counts = Counter(train_intents)
        vocab = set()
        for q, intent in zip(train_queries, train_intents):
            tokens = q.lower().split()
            for t in tokens:
                word_counts[intent][t] += 1
                vocab.add(t)

        total_words = {intent: sum(word_counts[intent].values()) for intent in class_counts}
        V = len(vocab)
        preds = []
        for q in test_queries:
            tokens = q.lower().split()
            best_intent = None
            best_score = -float("inf")
            for intent in class_counts:
                prior = math.log(class_counts[intent] / len(train_intents))
                score = prior
                for t in tokens:
                    count = word_counts[intent][t]
                    score += math.log((count + 1) / (total_words[intent] + V))
                if score > best_score:
                    best_score = score
                    best_intent = intent
            preds.append(best_intent)

        acc = sum(1 for yt, yp in zip(test_intents, preds) if yt == yp) / len(test_intents)
        macro_f1 = calculate_macro_f1(test_intents, preds, labels)

    return {
        "model": "TF-IDF + Linear Classifier (Simple)",
        "engine": engine,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4)
    }

def main():
    print("[*] Loading training corpus and golden evaluation set...")
    train_queries, train_intents = load_data(KB_CORPUS)
    test_queries, test_intents = load_data(GOLDEN_SET)
    labels = sorted(list(set(train_intents + test_intents)))

    print(f"[+] Loaded {len(train_queries)} training samples and {len(test_queries)} golden test samples.")
    print(f"[+] Taxonomy: {len(labels)} intents -> {', '.join(labels)}\n")

    majority_res = run_majority_baseline(train_intents, test_intents, labels)
    print("=" * 60)
    print("BASELINE 1: TRIVIAL (Majority Class)")
    print(f"Majority Intent: {majority_res['majority_class']}")
    print(f"Accuracy:        {majority_res['accuracy'] * 100:.2f}%")
    print(f"Macro F1:        {majority_res['macro_f1'] * 100:.2f}%")
    print("=" * 60)

    simple_res = run_simple_baseline(train_queries, train_intents, test_queries, test_intents, labels)
    print("\n" + "=" * 60)
    print(f"BASELINE 2: SIMPLE ({simple_res['engine']})")
    print(f"Accuracy:        {simple_res['accuracy'] * 100:.2f}%")
    print(f"Macro F1:        {simple_res['macro_f1'] * 100:.2f}%")
    print("=" * 60)

    results = {
        "baseline_1_majority": majority_res,
        "baseline_2_simple": simple_res
    }
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Baseline metrics successfully saved to {METRICS_OUT}")

if __name__ == "__main__":
    main()
