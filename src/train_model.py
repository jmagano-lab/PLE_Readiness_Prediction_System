"""
PLE Readiness Prediction System - Training Pipeline

This script trains and evaluates classification models that predict readiness level:
- Ready
- Moderately Ready
- At Risk

Inputs:
    data/source_uploaded_dataset.xlsx
Outputs:
    data/ple_readiness_cleaned.csv
    reports/model_leaderboard.csv
    reports/classification_report.txt
    models/best_readiness_model.joblib


"""

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "source_uploaded_dataset.xlsx"
RANDOM_STATE = 42

FEATURES = [
    "diagnostic", "pre_test", "post_test", "progress_evaluation", "mock_boards",
    "early_score", "later_score", "improvement_index", "mock_to_diagnostic_gap",
    "assessment_consistency"
]


def label_readiness(total_average: float) -> str:
    """Convert total average to a three-class readiness label."""
    if total_average >= 75:
        return "Ready"
    if total_average >= 50:
        return "Moderately Ready"
    return "At Risk"


def load_and_clean_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the OVERALL RESULT sheet and create modeling features."""
    raw = pd.read_excel(path, sheet_name="OVERALL RESULT")
    raw.columns = [str(c).strip() for c in raw.columns]
    df = raw.rename(columns={
        "STUDENT NAME": "student_name",
        "DIAGNOSTIC EXAMINATION": "diagnostic",
        "PROGRESS EVALUATION": "progress_evaluation",
        "MOCK BOARDS": "mock_boards",
        "POST TEST": "post_test",
        "PRE TEST": "pre_test",
        "TOTAL AVERAGE": "total_average",
    })
    keep = ["student_name", "diagnostic", "progress_evaluation", "mock_boards", "post_test", "pre_test", "total_average"]
    df = df[keep].copy().dropna(subset=["student_name"]).reset_index(drop=True)
    df["student_id"] = [f"S{i+1:03d}" for i in range(len(df))]
    numeric = ["diagnostic", "pre_test", "post_test", "progress_evaluation", "mock_boards", "total_average"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["readiness_level"] = df["total_average"].apply(label_readiness)
    df["ready_binary"] = np.where(df["readiness_level"].eq("Ready"), 1, 0)
    df["early_score"] = df[["diagnostic", "pre_test"]].mean(axis=1)
    df["later_score"] = df[["post_test", "progress_evaluation", "mock_boards"]].mean(axis=1)
    df["improvement_index"] = df["later_score"] - df["early_score"]
    df["mock_to_diagnostic_gap"] = df["mock_boards"] - df["diagnostic"]
    df["assessment_consistency"] = df[["diagnostic", "pre_test", "post_test", "progress_evaluation", "mock_boards"]].std(axis=1)
    return df


def get_models() -> dict:
    """Return candidate algorithms for comparison."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=400, max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
        "XGBoost (fallback: Gradient Boosting)": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    }


def train_and_evaluate() -> None:
    """Train models, compare metrics, and save the best model."""
    df = load_and_clean_data()
    (BASE / "data").mkdir(exist_ok=True)
    (BASE / "reports").mkdir(exist_ok=True)
    (BASE / "models").mkdir(exist_ok=True)
    df[["student_id"] + FEATURES + ["total_average", "readiness_level", "ready_binary"]].to_csv(BASE / "data" / "ple_readiness_cleaned.csv", index=False)

    X, y = df[FEATURES], df["readiness_level"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    preprocess = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), FEATURES)
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    leaderboard, trained = [], {}

    for name, clf in get_models().items():
        pipe = Pipeline([("preprocess", preprocess), ("model", clf)])
        scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro", "precision_macro": "precision_macro", "recall_macro": "recall_macro", "roc_auc_ovr": "roc_auc_ovr"}
        cvres = cross_validate(pipe, X, y, cv=cv, scoring=scoring, error_score=np.nan)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
        leaderboard.append({
            "model": name,
            "cv_accuracy_mean": np.nanmean(cvres["test_accuracy"]),
            "cv_f1_macro_mean": np.nanmean(cvres["test_f1_macro"]),
            "holdout_accuracy": accuracy_score(y_test, pred),
            "holdout_precision_macro": precision,
            "holdout_recall_macro": recall,
            "holdout_f1_macro": f1,
            "holdout_auc_ovr": roc_auc_score(y_test, proba, multi_class="ovr", labels=pipe.classes_),
        })
        trained[name] = pipe

    metrics = pd.DataFrame(leaderboard).sort_values(["holdout_f1_macro", "cv_f1_macro_mean"], ascending=False)
    metrics.to_csv(BASE / "reports" / "model_leaderboard.csv", index=False)
    best_name = metrics.iloc[0]["model"]
    best_model = trained[best_name]
    joblib.dump(best_model, BASE / "models" / "best_readiness_model.joblib")

    pred = best_model.predict(X_test)
    (BASE / "reports" / "classification_report.txt").write_text(classification_report(y_test, pred, zero_division=0))
    np.savetxt(BASE / "reports" / "confusion_matrix.csv", confusion_matrix(y_test, pred, labels=best_model.classes_), delimiter=",", fmt="%d")
    summary = {
        "n_students": int(len(df)),
        "class_distribution": df["readiness_level"].value_counts().to_dict(),
        "best_model": best_name,
        "leakage_note": "The label is derived from total_average, so external PLE outcome validation is required."
    }
    (BASE / "reports" / "project_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    train_and_evaluate()
