from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import shap


# ==============================================================
# PATHS
# ==============================================================

TRAIN_PATH = Path(
    "data/processed/ml/splits/train.csv"
)

TEST_PATH = Path(
    "data/processed/ml/splits/test.csv"
)

MODEL_SELECTION_PATH = Path(
    "models/model_selection.json"
)

PREPROCESSOR_PATH = Path(
    "models/preprocessor.joblib"
)

OUTPUT_DIR = Path(
    "data/processed/ml/explainability"
)


# ==============================================================
# FEATURE CONFIGURATION
# ==============================================================

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

FEATURE_COLUMNS = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


# ==============================================================
# LOAD ARTIFACTS
# ==============================================================

def load_artifacts():

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Testing data not found: {TEST_PATH}"
        )

    # Explain whichever model training selected, not a fixed
    # one, so SHAP output always matches the served model.
    with open(
        MODEL_SELECTION_PATH,
        encoding="utf-8",
    ) as file:
        selected_model = json.load(file)["selected_model"]

    model_path = Path(
        f"models/{selected_model}.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    print(f"Explaining selected model: {selected_model}")

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {PREPROCESSOR_PATH}"
        )

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    test_df = pd.read_csv(
        TEST_PATH
    )

    model = joblib.load(
        model_path
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    return (
        train_df,
        test_df,
        model,
        preprocessor,
    )


# ==============================================================
# RISK LEVEL
# ==============================================================

def get_risk_level(
    probability: float,
) -> str:

    if probability >= 0.60:
        return "HIGH"

    if probability >= 0.30:
        return "MEDIUM"

    return "LOW"


# ==============================================================
# MAP MODEL FEATURE TO BUSINESS FEATURE
# ==============================================================

def business_feature_name(
    transformed_feature_name: str,
) -> str:

    # Remove transformer prefixes.
    name = transformed_feature_name

    if "__" in name:
        name = name.split(
            "__",
            1,
        )[1]

    # Handle one-hot encoded categorical values.
    for category in CATEGORICAL_FEATURES:

        prefix = f"{category}_"

        if name.startswith(prefix):
            return category.replace(
                "_",
                " ",
            ).title()

    # Numeric feature.
    return name.replace(
        "_",
        " ",
    ).title()


# ==============================================================
# BUSINESS EXPLANATION
# ==============================================================

def generate_business_factor(
    feature_name: str,
    raw_row: pd.Series,
) -> str | None:

    feature = feature_name.lower()

    # ----------------------------------------------------------
    # Payment failure
    # ----------------------------------------------------------

    if feature == "payment_failure_rate":

        value = float(
            raw_row["payment_failure_rate"]
        )

        if value >= 0.15:
            return (
                f"High payment failure rate "
                f"({value:.1%})"
            )

    # ----------------------------------------------------------
    # Overdue amount
    # ----------------------------------------------------------

    if feature == "overdue_amount":

        value = float(
            raw_row["overdue_amount"]
        )

        if value > 0:
            return (
                f"Overdue amount of "
                f"{value:,.2f}"
            )

    # ----------------------------------------------------------
    # Overdue transactions
    # ----------------------------------------------------------

    if feature == "overdue_transaction_count":

        value = int(
            raw_row["overdue_transaction_count"]
        )

        if value > 0:
            return (
                f"{value} overdue transaction"
                + (
                    "s"
                    if value != 1
                    else ""
                )
            )

    # ----------------------------------------------------------
    # Failed payments
    # ----------------------------------------------------------

    if feature == "failed_transaction_count":

        value = int(
            raw_row["failed_transaction_count"]
        )

        if value > 0:
            return (
                f"{value} failed payment"
                + (
                    "s"
                    if value != 1
                    else ""
                )
            )

    # ----------------------------------------------------------
    # Payment delay
    # ----------------------------------------------------------

    if feature == "average_payment_delay":

        value = float(
            raw_row["average_payment_delay"]
        )

        if value > 0:
            return (
                f"Average payment delay of "
                f"{value:.1f} days"
            )

    # ----------------------------------------------------------
    # Complaints
    # ----------------------------------------------------------

    if feature == "complaint_count":

        value = int(
            raw_row["complaint_count"]
        )

        if value > 0:
            return (
                f"{value} customer complaint"
                + (
                    "s"
                    if value != 1
                    else ""
                )
            )

    # ----------------------------------------------------------
    # Recent complaints
    # ----------------------------------------------------------

    if feature == "recent_complaint_count":

        value = int(
            raw_row["recent_complaint_count"]
        )

        if value > 0:
            return (
                f"{value} recent complaint"
                + (
                    "s"
                    if value != 1
                    else ""
                )
            )

    # ----------------------------------------------------------
    # Negative sentiment
    # ----------------------------------------------------------

    if feature == "negative_sentiment_count":

        value = int(
            raw_row["negative_sentiment_count"]
        )

        if value > 0:
            return (
                f"{value} negative customer interaction"
                + (
                    "s"
                    if value != 1
                    else ""
                )
            )

    # ----------------------------------------------------------
    # Escalations
    # ----------------------------------------------------------

    if feature == "escalation_count":

        value = int(
            raw_row["escalation_count"]
        )

        if value > 0:
            return (
                f"{value} escalation"
                + (
                    "s"
                    if value != 1
                    else ""
                )
            )

    # ----------------------------------------------------------
    # Activity decline
    # ----------------------------------------------------------

    if feature == "activity_change":

        value = float(
            raw_row["activity_change"]
        )

        if value < -0.20:
            return (
                f"Customer activity decreased "
                f"by {abs(value):.1%}"
            )

    # ----------------------------------------------------------
    # Pending transactions
    # ----------------------------------------------------------

    if feature == "pending_transaction_count":

        value = int(
            raw_row["pending_transaction_count"]
        )

        if value > 0:
            return (
                f"{value} pending transaction"
                + (
                    "s"
                    if value != 1
                    else ""
                )
            )

    return None


# ==============================================================
# MAIN EXPLAINABILITY PIPELINE
# ==============================================================

def main():

    print("\n" + "=" * 70)
    print("CUSTOMER RISK INTELLIGENCE")
    print("MODEL EXPLAINABILITY")
    print("=" * 70)

    (
        train_df,
        test_df,
        model,
        preprocessor,
    ) = load_artifacts()

    # ----------------------------------------------------------
    # Prepare features
    # ----------------------------------------------------------

    X_train = train_df[
        FEATURE_COLUMNS
    ].copy()

    X_test = test_df[
        FEATURE_COLUMNS
    ].copy()

    # ----------------------------------------------------------
    # Transform using the SAME fitted preprocessor.
    # ----------------------------------------------------------

    X_train_transformed = (
        preprocessor.transform(
            X_train
        )
    )

    X_test_transformed = (
        preprocessor.transform(
            X_test
        )
    )

    print(
        f"\nTraining transformed shape: "
        f"{X_train_transformed.shape}"
    )

    print(
        f"Testing transformed shape: "
        f"{X_test_transformed.shape}"
    )

    # ----------------------------------------------------------
    # Generate probabilities
    # ----------------------------------------------------------

    probabilities = (
        model.predict_proba(
            X_test_transformed
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.60
    ).astype(int)

    # ----------------------------------------------------------
    # Create SHAP explainer
    #
    # For Logistic Regression, SHAP can explain the contribution
    # of each transformed feature to the model prediction.
    # ----------------------------------------------------------

    print(
        "\nCreating SHAP explainer..."
    )

    background_size = min(
        200,
        X_train_transformed.shape[0],
    )

    background = (
        X_train_transformed[
            :background_size
        ]
    )

    explainer = shap.Explainer(
        model,
        background,
    )

    shap_result = explainer(
        X_test_transformed
    )

    shap_values = shap_result.values

    # ----------------------------------------------------------
    # Binary logistic regression SHAP output is normally:
    #
    # rows × transformed_features
    #
    # Handle an additional output dimension defensively.
    # ----------------------------------------------------------

    if shap_values.ndim == 3:

        shap_values = (
            shap_values[:, :, -1]
        )

    # ----------------------------------------------------------
    # Get transformed feature names.
    # ----------------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    # ----------------------------------------------------------
    # Build prediction records.
    # ----------------------------------------------------------

    output_records = []

    max_factor_count = 5

    for row_index in range(
        len(test_df)
    ):

        raw_row = test_df.iloc[
            row_index
        ]

        customer_id = raw_row[
            "customer_id"
        ]

        probability = float(
            probabilities[row_index]
        )

        risk_level = get_risk_level(
            probability
        )

        row_shap = (
            shap_values[row_index]
        )

        # ------------------------------------------------------
        # Most positive SHAP contributions.
        #
        # Positive contribution pushes the customer toward
        # high-risk.
        # ------------------------------------------------------

        ranked_indices = np.argsort(
            row_shap
        )[::-1]

        selected_factors = []

        for feature_index in ranked_indices:

            contribution = float(
                row_shap[feature_index]
            )

            if contribution <= 0:
                break

            transformed_name = (
                feature_names[
                    feature_index
                ]
            )

            business_name = (
                business_feature_name(
                    transformed_name
                )
            )

            business_factor = (
                generate_business_factor(
                    business_name
                    .lower()
                    .replace(
                        " ",
                        "_",
                    ),
                    raw_row,
                )
            )

            if business_factor is None:
                continue

            if business_factor in selected_factors:
                continue

            selected_factors.append(
                business_factor
            )

            if (
                len(selected_factors)
                >= max_factor_count
            ):
                break

        # ------------------------------------------------------
        # Generic fallback if SHAP doesn't map to a
        # business-readable explanation.
        # ------------------------------------------------------

        if not selected_factors:

            if (
                raw_row[
                    "overdue_transaction_count"
                ] > 0
            ):
                selected_factors.append(
                    "Overdue transactions"
                )

            if (
                raw_row[
                    "payment_failure_rate"
                ] > 0
            ):
                selected_factors.append(
                    "Payment failures"
                )

            if (
                raw_row[
                    "complaint_count"
                ] > 0
            ):
                selected_factors.append(
                    "Customer complaints"
                )

        output_records.append(
            {
                "customer_id": customer_id,
                "observation_date": raw_row[
                    "observation_date"
                ],
                "actual_target": int(
                    raw_row["target"]
                ),
                "predicted_risk_probability": round(
                    probability,
                    6,
                ),
                "risk_level": risk_level,
                "risk_factors": selected_factors,
            }
        )

    # ----------------------------------------------------------
    # Save predictions
    # ----------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df = pd.DataFrame(
        output_records
    )

    prediction_path = (
        OUTPUT_DIR
        / "customer_risk_predictions.csv"
    )

    predictions_df.to_csv(
        prediction_path,
        index=False,
    )

    # ----------------------------------------------------------
    # Save SHAP values
    # ----------------------------------------------------------

    shap_df = pd.DataFrame(
        shap_values,
        columns=feature_names,
    )

    shap_path = (
        OUTPUT_DIR
        / "shap_values.csv"
    )

    shap_df.to_csv(
        shap_path,
        index=False,
    )

    # ----------------------------------------------------------
    # Global feature importance
    # ----------------------------------------------------------

    mean_abs_shap = (
        np.abs(shap_values)
        .mean(axis=0)
    )

    global_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    importance_path = (
        OUTPUT_DIR
        / "global_shap_importance.csv"
    )

    global_importance.to_csv(
        importance_path,
        index=False,
    )

    # ----------------------------------------------------------
    # Save sample JSON
    # ----------------------------------------------------------

    sample_records = (
        output_records[:10]
    )

    sample_json_path = (
        OUTPUT_DIR
        / "sample_risk_predictions.json"
    )

    with open(
        sample_json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            sample_records,
            file,
            indent=4,
        )

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPLAINABILITY OUTPUT")
    print("=" * 70)

    print(
        f"\nPredictions: "
        f"{prediction_path}"
    )

    print(
        f"SHAP values: "
        f"{shap_path}"
    )

    print(
        f"Global SHAP importance: "
        f"{importance_path}"
    )

    print(
        f"Sample JSON: "
        f"{sample_json_path}"
    )

    print("\nTop global features:")

    print(
        global_importance.head(
            10
        ).to_string(
            index=False
        )
    )

    print("\nSample predictions:")

    print(
        predictions_df.head(
            10
        ).to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("MODEL EXPLAINABILITY COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
