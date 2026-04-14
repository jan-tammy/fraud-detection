# Fraud Detection using Machine Learning

End-to-end fraud detection pipeline on PaySim — featuring statistical EDA, 
cost-sensitive threshold optimization, SHAP explainability, and a Streamlit 
deployment interface.

---

## Problem Statement

Fraudulent transactions account for a small fraction of total volume but carry 
disproportionate financial impact. Standard classifiers optimized for accuracy fail 
under severe class imbalance. This project addresses that gap with a pipeline designed 
around precision-recall tradeoffs and cost-sensitive decision making.

---

## Dataset

- **Source:** [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) — a synthetic mobile money transaction simulator
- **Size:** ~6.3 million transactions
- **Fraud rate:** ~0.13% (highly imbalanced)
- **Features:** Transaction type, amount, origin/destination balances, time step

---

## Methodology

| Phase | Description |
|-------|-------------|
| 1. EDA | K-S tests, chi-square, point-biserial correlation, ACF/PACF |
| 2. Feature Engineering | Balance discrepancies, cyclical time encoding, VIF checks, leakage audit |
| 3. Model Selection | Hyperparameter Tuning using Bayesian Optimization (Optuna) |
| 4. Explainability | SHAP beeswarm, waterfall, dependence plots, fraud archetype clustering |
| 5. Threshold Optimization | Cost-sensitive cutoff selection (FN: $500, FP: $10) |

---

## Features Used

| Feature | Description |
|---------|-------------|
| `amt_zscore` | Standardized transaction amount relative to training distribution |
| `orig_bal_discrepancy` | Difference between expected and actual origin balance after transaction |
| `dest_bal_discrepancy` | Difference between expected and actual destination balance after transaction |
| `zeroed_out` | Flag — origin account balance reduced to zero |
| `orig_empty` | Flag — origin account was already empty before transaction |
| `hour_sin / hour_cos` | Cyclical encoding of transaction hour |
| `day_sin / day_cos` | Cyclical encoding of transaction day |
| `is_transfer` | Flag — transaction type is TRANSFER or CASH_OUT |

---

## Key Results

- **Primary metric:** PR-AUC (preferred over accuracy under class imbalance): 0.8237 on held out test set
- **Model:** XGBoost tuned via Bayesian Optimization
- **Threshold:** Selected to minimize total expected cost (FN: $500, FP: $10): 0.7171
- **Output:** Net financial savings estimate on held-out test set: $671,260.00

---

## How to Run

1. Clone the repository
2. Install dependencies: 'pip install -r requirements.txt'
3. Run 'fraud_detection.ipynb' and use Kaggle API keys to access the dataset in-notebook.
4. Run 'fraud_detection.ipynb top to bottom to train and export model artifacts.
5. Launch the app: 'streamlit run app.py'

---

## Requirements

See 'requirements.txt' for pinned versions. Core dependencies:

pandas · numpy · scikit-learn · xgboost · imbalanced-learn · optuna · shap · streamlit · joblib · scipy · matplotlib · seaborn

---

## Author

### Janhavi Tamhankar
[LinkedIn](https://www.linkedin.com/in/janhavitamhankar/) | [Github](https://github.com/jan-tammy)
