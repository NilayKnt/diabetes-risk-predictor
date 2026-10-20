"""Train the diabetes risk model and export it to the web page.

Pipeline: median imputation -> standardization -> logistic regression.
Hyperparameters are tuned with GridSearchCV (stratified 5-fold, ROC-AUC).

Outputs:
  model.json  - imputer medians, scaler stats, coefficients, intercept, metrics
  index.html  - the model is embedded between the MODEL_START / MODEL_END markers

Usage:
  python train.py
  python train.py --data data/diabetes_cleaned.csv --no-page
"""
import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# `penalty` is deprecated in scikit-learn >= 1.8 (removed in 1.10) but still works.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

RANDOM_STATE = 42
# In this dataset a value of 0 means "not measured" for these columns.
ZERO_MEANS_MISSING = ["Glucose", "BloodPressure"]


def load_data(path: Path):
    df = pd.read_csv(path)
    df[ZERO_MEANS_MISSING] = df[ZERO_MEANS_MISSING].replace(0, np.nan)
    X = df.drop(columns="Outcome")
    y = df["Outcome"]
    return X, y


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="liblinear", max_iter=2000)),
        ]
    )


PARAM_GRID = {
    "clf__C": [0.001, 0.01, 0.1, 0.3, 1, 3, 10, 100],
    "clf__penalty": ["l1", "l2"],
    "clf__class_weight": [None, "balanced"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/diabetes_cleaned.csv")
    parser.add_argument("--out", default="model.json")
    parser.add_argument("--page", default="index.html")
    parser.add_argument("--no-page", action="store_true", help="do not update index.html")
    args = parser.parse_args()

    X, y = load_data(Path(args.data))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # 1) Hyperparameter search with cross-validation on the training split
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(build_pipeline(), PARAM_GRID, cv=cv, scoring="roc_auc", n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best params:     {search.best_params_}")

    # 2) Hold-out test set
    pred = search.predict(X_test)
    proba = search.predict_proba(X_test)[:, 1]
    test = {
        "auc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "precision": precision_score(y_test, pred),
    }
    print("Test set:", {k: round(v, 4) for k, v in test.items()})

    # 3) Steadier estimate: repeated CV on all data with the chosen settings
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
    rep_auc = cross_val_score(search.best_estimator_, X, y, cv=rcv, scoring="roc_auc")
    rep_acc = cross_val_score(search.best_estimator_, X, y, cv=rcv, scoring="accuracy")
    print(f"Repeated CV ROC-AUC:  {rep_auc.mean():.4f} +/- {rep_auc.std():.4f}")
    print(f"Repeated CV accuracy: {rep_acc.mean():.4f} +/- {rep_acc.std():.4f}")

    # 4) Final model on all data, exported for the browser
    final = clone(search.best_estimator_).fit(X, y)
    imputer, scaler, clf = final["imputer"], final["scaler"], final["clf"]
    model = {
        "features": list(X.columns),
        "median": imputer.statistics_.tolist(),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "min": X.min().tolist(),
        "max": X.max().tolist(),
        "params": {
            "penalty": search.best_params_["clf__penalty"],
            "C": search.best_params_["clf__C"],
            "class_weight": search.best_params_["clf__class_weight"],
        },
        "metrics": {
            "n_rows": int(len(X)),
            "positive_rate": float(y.mean()),
            "cv_auc": float(search.best_score_),
            "repeated_auc_mean": float(rep_auc.mean()),
            "repeated_auc_std": float(rep_auc.std()),
            "repeated_acc_mean": float(rep_acc.mean()),
            "repeated_acc_std": float(rep_acc.std()),
            "test_auc": float(test["auc"]),
            "test_recall": float(test["recall"]),
        },
    }
    Path(args.out).write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")

    if not args.no_page:
        page = Path(args.page)
        html = page.read_text(encoding="utf-8")
        pattern = re.compile(r"/\*MODEL_START\*/.*?/\*MODEL_END\*/", re.S)
        if not pattern.search(html):
            raise SystemExit(f"Markers not found in {page}")
        html = pattern.sub(lambda _: "/*MODEL_START*/" + json.dumps(model) + "/*MODEL_END*/", html)
        page.write_text(html, encoding="utf-8")
        print(f"Updated {page}")


if __name__ == "__main__":
    main()
