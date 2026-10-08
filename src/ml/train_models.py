from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from src.config.settings import settings


# ==============================================================
# PATHS
# ==============================================================

TRAIN_PATH = Path(
    "data/processed/ml/splits/train.csv"
)

TEST_PATH = Path(
    "data/processed/ml/splits/test.csv"
)

MODEL_DIR = Path("models")

METRICS_DIR = Path(
    "data/processed/ml"
)


# ==============================================================
# DATASET CONFIGURATION
# ==============================================================

TARGET_COLUMN = "target"

# PR-AUC differences below this are treated as a tie during
# model selection.
PR_AUC_TOLERANCE = 0.01

ID_COLUMNS = [
    "customer_id",
    "observation_date",
]

CATEGORICAL_FEATURES = [
    "customer_segment",
    "industry",
    "country",
    "account_status",
]

NUMERIC_FEATURES = [
    "customer_tenure_days",
    "total_transaction_amount",
    "average_transaction_amount",
    "transaction_count",
    "overdue_transaction_count",
    "overdue_amount",
    "failed_transaction_count",
    "payment_failure_rate",
    "average_payment_delay",
    "transaction_frequency",
    "recent_transaction_count",
    "previous_transaction_count",
    "activity_change",
    "pending_transaction_count",
    "interaction_count",
    "complaint_count",
    "support_interaction_count",
    "negative_sentiment_count",
    "escalation_count",
    "recent_complaint_count",
    "recent_negative_sentiment_count",
    "document_count",
    "complaint_document_count",
    "support_document_count",
]


# ==============================================================
# DATA LOADING
# ==============================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Testing dataset not found: {TEST_PATH}"
        )

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    test_df = pd.read_csv(
        TEST_PATH
    )

    return train_df, test_df


# ==============================================================
# FEATURE VALIDATION
# ==============================================================

def validate_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:

    expected_features = (
        CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
    )

    required_columns = (
        expected_features
        + [TARGET_COLUMN]
    )

    for column in required_columns:

        if column not in train_df.columns:
            raise ValueError(
                f"Missing column in training data: {column}"
            )

        if column not in test_df.columns:
            raise ValueError(
                f"Missing column in testing data: {column}"
            )

    # ----------------------------------------------------------
    # Ensure no audit/future columns accidentally entered.
    # ----------------------------------------------------------

    forbidden_columns = {
        "future_transaction_count",
        "future_failed_count",
        "future_overdue_count",
        "future_overdue_amount",
        "future_complaint_count",
        "future_negative_count",
        "future_escalation_count",
        "risk_score",
    }

    leaked_columns = (
        forbidden_columns
        & set(train_df.columns)
    )

    if leaked_columns:
        raise ValueError(
            "Potential target leakage detected. "
            f"Forbidden columns found: {sorted(leaked_columns)}"
        )

    # ----------------------------------------------------------
    # Verify feature lists are actually represented in dataset.
    # ----------------------------------------------------------

    feature_columns = (
        CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
    )

    missing_features = (
        set(feature_columns)
        - set(train_df.columns)
    )

    if missing_features:
        raise ValueError(
            f"Missing model features: "
            f"{sorted(missing_features)}"
        )


# ==============================================================
# PREPROCESSOR
# ==============================================================

def create_preprocessor() -> ColumnTransformer:

    # ----------------------------------------------------------
    # Numerical pipeline
    #
    # Median imputation protects us if future data contains
    # missing numeric values.
    # ----------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(
                    with_mean=False
                ),
            ),
        ]
    )

    # ----------------------------------------------------------
    # Categorical pipeline
    #
    # Unknown categories are ignored instead of crashing.
    # ----------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


# ==============================================================
# METRICS
# ==============================================================

def evaluate_model(
    model_name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    tn, fp, fn, tp = matrix.ravel()

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    return metrics


# ==============================================================
# PRINT METRICS
# ==============================================================

def print_model_metrics(
    metrics: dict,
) -> None:

    print("\n" + "-" * 70)

    print(
        f"MODEL: {metrics['model']}"
    )

    print("-" * 70)

    print(
        f"Accuracy : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {metrics['recall']:.4f}"
    )

    print(
        f"F1       : {metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : {metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC   : {metrics['pr_auc']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        f"TN={metrics['true_negative']}, "
        f"FP={metrics['false_positive']}, "
        f"FN={metrics['false_negative']}, "
        f"TP={metrics['true_positive']}"
    )


# ==============================================================
# SAVE MODEL
# ==============================================================

def save_model(
    model,
    model_name: str,
) -> None:

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        MODEL_DIR
        / f"{model_name}.joblib"
    )

    joblib.dump(
        model,
        path,
    )

    print(
        f"Saved model: {path}"
    )


# ==============================================================
# SAVE METRICS
# ==============================================================

def save_metrics(
    metrics_list: list[dict],
) -> None:

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # CSV
    # ----------------------------------------------------------

    metrics_df = pd.DataFrame(
        metrics_list
    )

    csv_path = (
        METRICS_DIR
        / "model_comparison.csv"
    )

    metrics_df.to_csv(
        csv_path,
        index=False,
    )

    # ----------------------------------------------------------
    # JSON
    # ----------------------------------------------------------

    json_path = (
        METRICS_DIR
        / "model_metrics.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics_list,
            file,
            indent=4,
        )

    print(
        f"\nSaved metrics CSV: {csv_path}"
    )

    print(
        f"Saved metrics JSON: {json_path}"
    )


# ==============================================================
# MAIN
# ==============================================================

def main() -> None:

    print("\n" + "=" * 70)
    print("CUSTOMER RISK INTELLIGENCE")
    print("ML MODEL TRAINING")
    print("=" * 70)

    # ----------------------------------------------------------
    # Load datasets
    # ----------------------------------------------------------

    train_df, test_df = load_data()

    print(
        f"\nTraining rows: {len(train_df):,}"
    )

    print(
        f"Testing rows:  {len(test_df):,}"
    )

    # ----------------------------------------------------------
    # Validate features
    # ----------------------------------------------------------

    validate_features(
        train_df,
        test_df,
    )

    print(
        "\n[PASS] Feature validation"
    )

    # ----------------------------------------------------------
    # Select features
    # ----------------------------------------------------------

    feature_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    X_train = train_df[
        feature_columns
    ].copy()

    y_train = train_df[
        TARGET_COLUMN
    ].astype(int)

    X_test = test_df[
        feature_columns
    ].copy()

    y_test = test_df[
        TARGET_COLUMN
    ].astype(int)

    # ----------------------------------------------------------
    # Create preprocessing
    # ----------------------------------------------------------

    preprocessor = create_preprocessor()

    # ----------------------------------------------------------
    # Fit preprocessor only on TRAIN data.
    #
    # Important:
    # We never fit preprocessing using the test dataset.
    # ----------------------------------------------------------

    X_train_transformed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_test_transformed = (
        preprocessor.transform(
            X_test
        )
    )

    print(
        "\n[PASS] Preprocessor fitted only on training data"
    )

    print(
        f"Transformed training shape: "
        f"{X_train_transformed.shape}"
    )

    print(
        f"Transformed testing shape: "
        f"{X_test_transformed.shape}"
    )

    # ----------------------------------------------------------
    # Save preprocessor
    # ----------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    preprocessor_path = (
        MODEL_DIR
        / "preprocessor.joblib"
    )

    joblib.dump(
        preprocessor,
        preprocessor_path,
    )

    print(
        f"Saved preprocessor: "
        f"{preprocessor_path}"
    )

    # ==========================================================
    # MODEL 1 — MAJORITY CLASS BASELINE
    # ==========================================================

    print("\n" + "=" * 70)
    print("MODEL 1 — DUMMY BASELINE")
    print("=" * 70)

    dummy_model = DummyClassifier(
        strategy="most_frequent"
    )

    dummy_model.fit(
        X_train_transformed,
        y_train,
    )

    # DummyClassifier has predict_proba.
    dummy_metrics = evaluate_model(
        "dummy_baseline",
        dummy_model,
        X_test_transformed,
        y_test,
    )

    print_model_metrics(
        dummy_metrics
    )

    # ==========================================================
    # MODEL 2 — LOGISTIC REGRESSION
    # ==========================================================

    print("\n" + "=" * 70)
    print("MODEL 2 — LOGISTIC REGRESSION")
    print("=" * 70)

    logistic_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    logistic_model.fit(
        X_train_transformed,
        y_train,
    )

    logistic_metrics = evaluate_model(
        "logistic_regression",
        logistic_model,
        X_test_transformed,
        y_test,
    )

    print_model_metrics(
        logistic_metrics
    )

    save_model(
        logistic_model,
        "logistic_regression",
    )

    # ==========================================================
    # MODEL 3 — RANDOM FOREST
    # ==========================================================

    print("\n" + "=" * 70)
    print("MODEL 3 — RANDOM FOREST")
    print("=" * 70)

    random_forest_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    random_forest_model.fit(
        X_train_transformed,
        y_train,
    )

    random_forest_metrics = evaluate_model(
        "random_forest",
        random_forest_model,
        X_test_transformed,
        y_test,
    )

    print_model_metrics(
        random_forest_metrics
    )

    save_model(
        random_forest_model,
        "random_forest",
    )

    # ==========================================================
    # MODEL 4 — XGBOOST
    # ==========================================================

    print("\n" + "=" * 70)
    print("MODEL 4 — XGBOOST")
    print("=" * 70)

    positive_count = (
        y_train == 1
    ).sum()

    negative_count = (
        y_train == 0
    ).sum()

    scale_pos_weight = (
        negative_count / positive_count
        if positive_count > 0
        else 1.0
    )

    print(
        f"XGBoost scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    xgb_model.fit(
        X_train_transformed,
        y_train,
    )

    xgb_metrics = evaluate_model(
        "xgboost",
        xgb_model,
        X_test_transformed,
        y_test,
    )

    print_model_metrics(
        xgb_metrics
    )

    save_model(
        xgb_model,
        "xgboost",
    )

    # ==========================================================
    # MODEL COMPARISON
    # ==========================================================

    all_metrics = [
        dummy_metrics,
        logistic_metrics,
        random_forest_metrics,
        xgb_metrics,
    ]

    save_metrics(
        all_metrics
    )

    comparison_df = (
        pd.DataFrame(
            all_metrics
        )
        .sort_values(
            "pr_auc",
            ascending=False,
        )
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        comparison_df[
            [
                "model",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
            ]
        ].to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # Select final candidate.
    #
    # PR-AUC first, but differences smaller than
    # PR_AUC_TOLERANCE are treated as a tie (on ~1.8k test rows
    # a 0.002 gap is noise). Ties are broken by recall - missing
    # a high-risk customer is the more expensive error - then F1.
    # ----------------------------------------------------------

    candidate_df = comparison_df[
        comparison_df["model"]
        != "dummy_baseline"
    ].copy()

    best_pr_auc = candidate_df["pr_auc"].max()

    shortlist = candidate_df[
        candidate_df["pr_auc"]
        >= best_pr_auc - PR_AUC_TOLERANCE
    ]

    final_row = (
        shortlist
        .sort_values(
            [
                "recall",
                "f1",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    final_model_name = final_row["model"]

    print("\n" + "=" * 70)
    print("FINAL MODEL CANDIDATE")
    print("=" * 70)

    print(
        f"Selected model: "
        f"{final_model_name}"
    )

    print(
        f"Shortlist (PR-AUC within {PR_AUC_TOLERANCE} of "
        f"best {best_pr_auc:.4f}): "
        f"{shortlist['model'].tolist()}"
    )

    print(
        "\nSelection priority:"
    )

    print(
        f"1. PR-AUC (ties within {PR_AUC_TOLERANCE})"
    )

    print(
        "2. Recall"
    )

    print(
        "3. F1"
    )

    print(
        "\nReason:"
    )

    print(
        "High-risk detection is an imbalanced classification "
        "problem, so PR-AUC and recall are more informative "
        "than accuracy alone."
    )

    # ----------------------------------------------------------
    # Save selection metadata
    # ----------------------------------------------------------

    selection_path = (
        MODEL_DIR
        / "model_selection.json"
    )

    selection_data = {
        "selected_model": final_model_name,
        "model_version": "v1",
        "selection_criteria": [
            "pr_auc",
            "recall",
            "f1",
        ],
        "pr_auc_tolerance": PR_AUC_TOLERANCE,
        "shortlist": shortlist["model"].tolist(),
        "test_metrics": {
            metric: round(float(final_row[metric]), 4)
            for metric in [
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
            ]
        },
        "high_risk_threshold": settings.HIGH_RISK_THRESHOLD,
    }

    with open(
        selection_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            selection_data,
            file,
            indent=4,
        )

    print(
        f"\nSaved model selection: "
        f"{selection_path}"
    )

    print("\n" + "=" * 70)
    print("ML MODEL TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
