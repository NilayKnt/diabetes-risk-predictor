# Diabetes Risk Predictor

A logistic regression model trained on the Pima Indians Diabetes dataset, with a single-file web page that runs the model in the browser. No server, no data leaves the page.

**Live demo:** `https://nilayknt.github.io/diabetes-risk-predictor/` (after you enable GitHub Pages, see below)

> **Not medical advice.** This project was built for machine learning education. The dataset covers women of Pima Indian heritage aged 21 and over, so results may not apply to other groups.

## Results

Cleaned dataset: 768 rows, 8 inputs, 34.9% positive class.

| Metric | Value |
| --- | --- |
| Best CV ROC-AUC (5-fold, grid search on 80% train split) | 0.846 |
| Repeated CV ROC-AUC (5x5, all data) | 0.837 ± 0.028 |
| Repeated CV accuracy (5x5, all data) | 0.753 ± 0.033 |
| Hold-out test set (20%) | AUC 0.809, accuracy 0.727, recall 0.704, precision 0.594 |

Selected hyperparameters: L1 penalty, `C=0.3`, `class_weight="balanced"`.

Because the class weights are balanced, the score is a model output tuned to catch the positive class, not a calibrated probability.

## How it works

1. Zeros in `Glucose` and `BloodPressure` are treated as missing values (in this dataset 0 means "not measured").
2. Pipeline: median imputation, standardization, logistic regression.
3. `GridSearchCV` with `StratifiedKFold(5)` tunes `C`, `penalty` and `class_weight`, scored by ROC-AUC.
4. The final model is refit on all data. Its imputer medians, scaler statistics, coefficients and intercept are exported to `model.json` and embedded in `index.html`.
5. The page applies the same math in JavaScript: impute, standardize, `sigmoid(intercept + sum(coef * z))`.

## Project structure

```
.
├── data/
│   └── diabetes_cleaned.csv   # cleaned dataset (missing values left empty)
├── index.html                 # the web page (model embedded)
├── model.json                 # exported model and metrics
├── train.py                   # training, cross-validation, export
├── requirements.txt
└── README.md
```

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py
```

`train.py` prints the scores, rewrites `model.json`, and updates the model embedded in `index.html`. Then open `index.html` in a browser.

Options:

```bash
python train.py --data data/diabetes_cleaned.csv --no-page   # do not touch index.html
```

## Notes

- `requirements.txt` limits scikit-learn to `<1.10` because the `penalty` argument of `LogisticRegression` is deprecated in 1.8 and removed in 1.10.
- Inputs outside the training range show a warning, because linear extrapolation can be unreliable there.
- Next steps: a FastAPI service and MLflow experiment tracking.

## Data

Pima Indians Diabetes Database, originally from the National Institute of Diabetes and Digestive and Kidney Diseases.
