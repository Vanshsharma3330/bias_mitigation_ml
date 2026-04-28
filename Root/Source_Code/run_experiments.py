import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import load_adult, load_compas
from src.preprocessing import encode
from src.evaluation import evaluate
from src.fairness_metrics import demographic_parity, equal_opportunity
from src.visualization import plot_tradeoff, plot_bar
from src.bias_mitigation import (
    to_aif,
    reweigh,
    di_remover,
    prejudice_remover,
    adversarial_debiasing,
    eq_odds,
    calibrated_eq_odds,
    create_prediction_dataset,
)


DATASETS = [
    {
        "name": "Adult",
        "loader": load_adult,
        "target": "income",
        "protected": "sex",
        "privileged_value": 1,
    },
    {
        "name": "COMPAS",
        "loader": load_compas,
        "target": "two_year_recid",
        "protected": "race",
        "privileged_value": 0,
    },
]

MODEL_CONFIGS = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {"C": [0.1, 1.0, 10.0]},
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(),
        "params": {"max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]},
    },
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
    },
}


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        minv, maxv = raw.min(), raw.max()
        if maxv > minv:
            return (raw - minv) / (maxv - minv)
        return raw
    return model.predict(X)


def format_pred_labels(pred_dataset):
    return np.asarray(pred_dataset.labels).ravel()


def run_dataset(config):
    df = config["loader"]()
    df = encode(df)

    X = df.drop(config["target"], axis=1)
    y = df[config["target"]]
    p = df[config["protected"]]

    X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
        X,
        y,
        p,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    train_df = X_train.copy()
    train_df[config["target"]] = y_train
    train_df[config["protected"]] = p_train

    test_df = X_test.copy()
    test_df[config["target"]] = y_test
    test_df[config["protected"]] = p_test

    train_dataset = to_aif(
        train_df,
        config["target"],
        config["protected"],
        privileged_value=config["privileged_value"],
    )
    test_dataset = to_aif(
        test_df,
        config["target"],
        config["protected"],
        privileged_value=config["privileged_value"],
    )

    results = []
    labels = []

    print(f"\n=== Dataset: {config['name']} ===")

    for name, config_model in MODEL_CONFIGS.items():
        grid = GridSearchCV(config_model["model"], config_model["params"], cv=5, scoring="accuracy")
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        acc, f1 = evaluate(y_test, y_pred)
        dpd = demographic_parity(y_pred, p_test.values)
        eod = equal_opportunity(y_test.values, y_pred, p_test.values)

        results.append({"method": f"Baseline ({name})", "accuracy": acc, "f1": f1, "dpd": dpd, "eod": eod})
        labels.append(f"Baseline ({name})")
        print(name, "Baseline", f"acc={acc:.4f}", f"f1={f1:.4f}", f"dpd={dpd:.4f}", f"eod={eod:.4f}")

    rw_dataset = reweigh(train_dataset, privileged_value=config["privileged_value"])
    rw_df, _ = rw_dataset.convert_to_dataframe()
    w = rw_dataset.instance_weights
    X_rw = rw_df.drop(config["target"], axis=1)
    y_rw = rw_df[config["target"]]

    di_dataset = di_remover(train_dataset)
    di_df, _ = di_dataset.convert_to_dataframe()
    X_di = di_df.drop(config["target"], axis=1)
    y_di = di_df[config["target"]]

    for name, config_model in MODEL_CONFIGS.items():
        grid = GridSearchCV(config_model["model"], config_model["params"], cv=5, scoring="accuracy")
        grid.fit(X_rw, y_rw, sample_weight=w)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        acc, f1 = evaluate(y_test, y_pred)
        dpd = demographic_parity(y_pred, p_test.values)
        eod = equal_opportunity(y_test.values, y_pred, p_test.values)
        results.append({"method": f"Reweighing ({name})", "accuracy": acc, "f1": f1, "dpd": dpd, "eod": eod})
        labels.append(f"Reweighing ({name})")
        print(name, "Reweighing", f"acc={acc:.4f}", f"f1={f1:.4f}", f"dpd={dpd:.4f}", f"eod={eod:.4f}")

        grid = GridSearchCV(config_model["model"], config_model["params"], cv=5, scoring="accuracy")
        grid.fit(X_di, y_di)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        acc, f1 = evaluate(y_test, y_pred)
        dpd = demographic_parity(y_pred, p_test.values)
        eod = equal_opportunity(y_test.values, y_pred, p_test.values)
        results.append({"method": f"DI Remover ({name})", "accuracy": acc, "f1": f1, "dpd": dpd, "eod": eod})
        labels.append(f"DI Remover ({name})")
        print(name, "DI Remover", f"acc={acc:.4f}", f"f1={f1:.4f}", f"dpd={dpd:.4f}", f"eod={eod:.4f}")

    print("Training Prejudice Remover and Adversarial Debiasing...")
    pr_pred = prejudice_remover(train_dataset, test_dataset)
    y_pr = format_pred_labels(pr_pred)
    acc, f1 = evaluate(y_test, y_pr)
    dpd = demographic_parity(y_pr, p_test.values)
    eod = equal_opportunity(y_test.values, y_pr, p_test.values)
    results.append({"method": "Prejudice Remover", "accuracy": acc, "f1": f1, "dpd": dpd, "eod": eod})
    labels.append("Prejudice Remover")
    print("Prejudice Remover", f"acc={acc:.4f}", f"f1={f1:.4f}", f"dpd={dpd:.4f}", f"eod={eod:.4f}")

    try:
        adv_pred = adversarial_debiasing(train_dataset, test_dataset, privileged_value=config["privileged_value"], num_epochs=25)
        y_adv = format_pred_labels(adv_pred)
        acc, f1 = evaluate(y_test, y_adv)
        dpd = demographic_parity(y_adv, p_test.values)
        eod = equal_opportunity(y_test.values, y_adv, p_test.values)
        results.append({"method": "Adversarial Debiasing", "accuracy": acc, "f1": f1, "dpd": dpd, "eod": eod})
        labels.append("Adversarial Debiasing")
        print("Adversarial Debiasing", f"acc={acc:.4f}", f"f1={f1:.4f}", f"dpd={dpd:.4f}", f"eod={eod:.4f}")
    except ImportError as e:
        print(f"Adversarial Debiasing skipped: {e}")
        results.append({"method": "Adversarial Debiasing", "accuracy": float('nan'), "f1": float('nan'), "dpd": float('nan'), "eod": float('nan')})
        labels.append("Adversarial Debiasing")

    for name, config_model in MODEL_CONFIGS.items():
        grid = GridSearchCV(config_model["model"], config_model["params"], cv=5, scoring="accuracy")
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        train_scores = get_scores(best_model, X_train)
        train_pred = best_model.predict(X_train)
        test_scores = get_scores(best_model, X_test)
        test_pred = best_model.predict(X_test)

        train_pred_dataset = create_prediction_dataset(train_dataset, train_pred, train_scores)
        test_pred_dataset = create_prediction_dataset(test_dataset, test_pred, test_scores)

        eq_pred = eq_odds(train_dataset, train_pred_dataset, test_pred_dataset, privileged_value=config["privileged_value"])
        y_eq = format_pred_labels(eq_pred)
        acc, f1 = evaluate(y_test, y_eq)
        dpd = demographic_parity(y_eq, p_test.values)
        eod = equal_opportunity(y_test.values, y_eq, p_test.values)
        results.append({"method": f"Equalized Odds ({name})", "accuracy": acc, "f1": f1, "dpd": dpd, "eod": eod})
        labels.append(f"Equalized Odds ({name})")
        print(name, "Equalized Odds", f"acc={acc:.4f}", f"f1={f1:.4f}", f"dpd={dpd:.4f}", f"eod={eod:.4f}")

        ceq_pred = calibrated_eq_odds(train_dataset, train_pred_dataset, test_pred_dataset, privileged_value=config["privileged_value"])
        y_ceq = format_pred_labels(ceq_pred)
        acc, f1 = evaluate(y_test, y_ceq)
        dpd = demographic_parity(y_ceq, p_test.values)
        eod = equal_opportunity(y_test.values, y_ceq, p_test.values)
        results.append({"method": f"Calibrated Eq. Odds ({name})", "accuracy": acc, "f1": f1, "dpd": dpd, "eod": eod})
        labels.append(f"Calibrated Eq. Odds ({name})")
        print(name, "Calibrated Eq. Odds", f"acc={acc:.4f}", f"f1={f1:.4f}", f"dpd={dpd:.4f}", f"eod={eod:.4f}")

    df_results = pd.DataFrame(results)
    summary_path = project_root / "outputs" / "tables" / f"{config['name']}_results.csv"
    df_results.to_csv(summary_path, index=False)
    print(f"Saved results to {summary_path}")

    plot_tradeoff(results, config['name'])
    plot_bar(results, labels, "dpd", project_root / "outputs" / "plots" / f"{config['name']}_bar.png")
    plot_bar(results, labels, "accuracy", project_root / "outputs" / "plots" / f"{config['name']}_accuracy.png")


if __name__ == "__main__":
    # Ensure output directories exist
    (project_root / "outputs" / "tables").mkdir(parents=True, exist_ok=True)
    (project_root / "outputs" / "plots").mkdir(parents=True, exist_ok=True)
    
    for config in DATASETS:
        run_dataset(config)
