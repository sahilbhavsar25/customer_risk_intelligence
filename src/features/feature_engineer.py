from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import settings


# ==============================================================
# PATHS
# ==============================================================

PROCESSED_DIR = Path("data/processed")
ML_DIR = PROCESSED_DIR / "ml"


# ==============================================================
# FEATURE ENGINEERING
# ==============================================================

class CustomerFeatureEngineer:
    """
    Builds customer-level historical features and a
    future 30-day high-risk target.

    Important leakage rule:
    Features only use information available on or before
    the observation date.
    """

    def __init__(self):
        self.customers = self._load_csv(
            "customers",
            "customers_clean.csv",
        )

        self.transactions = self._load_csv(
            "transactions",
            "transactions_clean.csv",
        )

        self.interactions = self._load_csv(
            "interactions",
            "interactions_clean.csv",
        )

        self.documents = self._load_csv(
            "documents",
            "documents_clean.csv",
        )

        self._prepare_data()

    # ==========================================================
    # DATA LOADING
    # ==========================================================

    def _load_csv(
        self,
        folder_name: str,
        file_name: str,
    ) -> pd.DataFrame:

        path = (
            PROCESSED_DIR
            / folder_name
            / file_name
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}"
            )

        return pd.read_csv(path)

    # ==========================================================
    # DATA PREPARATION
    # ==========================================================

    def _prepare_data(self) -> None:

        # ------------------------------------------------------
        # Customer data
        # ------------------------------------------------------

        self.customers["customer_since"] = pd.to_datetime(
            self.customers["customer_since"],
            errors="coerce",
        )

        # ------------------------------------------------------
        # Transactions
        # ------------------------------------------------------

        self.transactions["transaction_date"] = pd.to_datetime(
            self.transactions["transaction_date"],
            errors="coerce",
        )

        self.transactions["due_date"] = pd.to_datetime(
            self.transactions["due_date"],
            errors="coerce",
        )

        self.transactions["payment_date"] = pd.to_datetime(
            self.transactions["payment_date"],
            errors="coerce",
        )

        self.transactions["transaction_amount"] = pd.to_numeric(
            self.transactions["transaction_amount"],
            errors="coerce",
        )

        # ------------------------------------------------------
        # Interactions
        # ------------------------------------------------------

        self.interactions["interaction_date"] = pd.to_datetime(
            self.interactions["interaction_date"],
            errors="coerce",
        )

        # ------------------------------------------------------
        # Documents
        # ------------------------------------------------------

        self.documents["document_date"] = pd.to_datetime(
            self.documents["document_date"],
            errors="coerce",
        )

        # ------------------------------------------------------
        # Create customer groups for faster access.
        # ------------------------------------------------------

        self.transaction_groups = {
            customer_id: group.sort_values(
                "transaction_date"
            ).copy()
            for customer_id, group
            in self.transactions.groupby("customer_id")
        }

        self.interaction_groups = {
            customer_id: group.sort_values(
                "interaction_date"
            ).copy()
            for customer_id, group
            in self.interactions.groupby("customer_id")
        }

        self.document_groups = {
            customer_id: group.sort_values(
                "document_date"
            ).copy()
            for customer_id, group
            in self.documents.groupby("customer_id")
        }

    # ==========================================================
    # OBSERVATION DATES
    # ==========================================================

    def get_observation_dates(self) -> list[pd.Timestamp]:
        """
        Create monthly observation dates.

        We need enough historical data for a 90-day feature
        window and enough future data for a 30-day target.
        """

        start_date = (
            pd.Timestamp(settings.DATA_START_DATE)
            + pd.Timedelta(
                days=settings.FEATURE_LOOKBACK_DAYS
            )
        )

        end_date = (
            pd.Timestamp(settings.DATA_END_DATE)
            - pd.Timedelta(
                days=settings.PREDICTION_WINDOW_DAYS
            )
        )

        dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq="ME",
        )

        return list(dates)

    # ==========================================================
    # TRANSACTION FEATURES
    # ==========================================================

    def _transaction_features(
        self,
        customer_id: str,
        observation_date: pd.Timestamp,
    ) -> dict:

        transactions = self.transaction_groups.get(
            customer_id,
            pd.DataFrame(),
        )

        if transactions.empty:
            return {
                "total_transaction_amount": 0.0,
                "average_transaction_amount": 0.0,
                "transaction_count": 0,
                "overdue_transaction_count": 0,
                "overdue_amount": 0.0,
                "failed_transaction_count": 0,
                "payment_failure_rate": 0.0,
                "average_payment_delay": 0.0,
                "transaction_frequency": 0.0,
                "recent_transaction_count": 0,
                "previous_transaction_count": 0,
                "activity_change": 0.0,
                "pending_transaction_count": 0,
            }

        lookback_start = (
            observation_date
            - pd.Timedelta(
                days=settings.FEATURE_LOOKBACK_DAYS - 1
            )
        )

        recent_start = (
            observation_date
            - pd.Timedelta(
                days=settings.RECENT_ACTIVITY_DAYS - 1
            )
        )

        previous_start = (
            observation_date
            - pd.Timedelta(
                days=(settings.RECENT_ACTIVITY_DAYS * 2) - 1
            )
        )

        # ------------------------------------------------------
        # Only transactions that already existed at prediction
        # time are allowed into the feature set.
        # ------------------------------------------------------

        historical = transactions[
            (transactions["transaction_date"] >= lookback_start)
            & (
                transactions["transaction_date"]
                <= observation_date
            )
        ].copy()

        if historical.empty:
            return {
                "total_transaction_amount": 0.0,
                "average_transaction_amount": 0.0,
                "transaction_count": 0,
                "overdue_transaction_count": 0,
                "overdue_amount": 0.0,
                "failed_transaction_count": 0,
                "payment_failure_rate": 0.0,
                "average_payment_delay": 0.0,
                "transaction_frequency": 0.0,
                "recent_transaction_count": 0,
                "previous_transaction_count": 0,
                "activity_change": 0.0,
                "pending_transaction_count": 0,
            }

        transaction_count = len(historical)

        total_amount = historical[
            "transaction_amount"
        ].sum()

        average_amount = historical[
            "transaction_amount"
        ].mean()

        # ------------------------------------------------------
        # Recent transaction activity.
        # ------------------------------------------------------

        recent = historical[
            historical["transaction_date"] >= recent_start
        ]

        previous = historical[
            historical["transaction_date"] >= previous_start
        ]

        previous = previous[
            previous["transaction_date"] < recent_start
        ]

        recent_count = len(recent)
        previous_count = len(previous)

        # ------------------------------------------------------
        # Activity change:
        #
        # Positive number = increased activity
        # Negative number = decreased activity
        # ------------------------------------------------------

        if previous_count > 0:
            activity_change = (
                recent_count - previous_count
            ) / previous_count
        elif recent_count > 0:
            activity_change = 1.0
        else:
            activity_change = 0.0

        # ------------------------------------------------------
        # Payment failures.
        # ------------------------------------------------------

        failed_count = (
            historical["payment_status"]
            .eq("FAILED")
            .sum()
        )

        payment_failure_rate = (
            failed_count / transaction_count
            if transaction_count > 0
            else 0.0
        )

        # ------------------------------------------------------
        # Determine overdue status AS OF observation date.
        #
        # We do not simply trust the final payment_status
        # because a payment could happen after our observation
        # date.
        # ------------------------------------------------------

        overdue_mask = (
            (historical["due_date"] < observation_date)
            & (
                historical["payment_date"].isna()
                | (
                    historical["payment_date"]
                    > observation_date
                )
            )
        )

        overdue_transactions = historical[
            overdue_mask
        ]

        overdue_count = len(
            overdue_transactions
        )

        overdue_amount = (
            overdue_transactions[
                "transaction_amount"
            ].sum()
            if overdue_count > 0
            else 0.0
        )

        # ------------------------------------------------------
        # Pending transactions AS OF observation date.
        # ------------------------------------------------------

        pending_mask = (
            (historical["due_date"] >= observation_date)
            & historical["payment_date"].isna()
            & (
                historical["payment_status"]
                != "FAILED"
            )
        )

        pending_count = pending_mask.sum()

        # ------------------------------------------------------
        # Payment delay.
        #
        # Only payments completed by the observation date
        # are included.
        # ------------------------------------------------------

        paid_by_observation = historical[
            historical["payment_date"].notna()
            & (
                historical["payment_date"]
                <= observation_date
            )
        ].copy()

        if not paid_by_observation.empty:
            payment_delay_days = (
                paid_by_observation["payment_date"]
                - paid_by_observation["due_date"]
            ).dt.days.clip(lower=0)

            average_payment_delay = (
                payment_delay_days.mean()
            )
        else:
            average_payment_delay = 0.0

        # ------------------------------------------------------
        # Transaction frequency normalized to monthly frequency.
        # ------------------------------------------------------

        transaction_frequency = (
            transaction_count / 3.0
        )

        return {
            "total_transaction_amount": float(
                total_amount
            ),
            "average_transaction_amount": float(
                average_amount
            ),
            "transaction_count": int(
                transaction_count
            ),
            "overdue_transaction_count": int(
                overdue_count
            ),
            "overdue_amount": float(
                overdue_amount
            ),
            "failed_transaction_count": int(
                failed_count
            ),
            "payment_failure_rate": float(
                payment_failure_rate
            ),
            "average_payment_delay": float(
                average_payment_delay
            ),
            "transaction_frequency": float(
                transaction_frequency
            ),
            "recent_transaction_count": int(
                recent_count
            ),
            "previous_transaction_count": int(
                previous_count
            ),
            "activity_change": float(
                activity_change
            ),
            "pending_transaction_count": int(
                pending_count
            ),
        }

    # ==========================================================
    # INTERACTION FEATURES
    # ==========================================================

    def _interaction_features(
        self,
        customer_id: str,
        observation_date: pd.Timestamp,
    ) -> dict:

        interactions = self.interaction_groups.get(
            customer_id,
            pd.DataFrame(),
        )

        if interactions.empty:
            return {
                "interaction_count": 0,
                "complaint_count": 0,
                "support_interaction_count": 0,
                "negative_sentiment_count": 0,
                "escalation_count": 0,
                "recent_complaint_count": 0,
                "recent_negative_sentiment_count": 0,
            }

        lookback_start = (
            observation_date
            - pd.Timedelta(
                days=settings.FEATURE_LOOKBACK_DAYS - 1
            )
        )

        recent_start = (
            observation_date
            - pd.Timedelta(
                days=settings.RECENT_ACTIVITY_DAYS - 1
            )
        )

        historical = interactions[
            (interactions["interaction_date"] >= lookback_start)
            & (
                interactions["interaction_date"]
                <= observation_date
            )
        ].copy()

        if historical.empty:
            return {
                "interaction_count": 0,
                "complaint_count": 0,
                "support_interaction_count": 0,
                "negative_sentiment_count": 0,
                "escalation_count": 0,
                "recent_complaint_count": 0,
                "recent_negative_sentiment_count": 0,
            }

        complaint_count = (
            historical["interaction_type"]
            .eq("COMPLAINT")
            .sum()
        )

        support_count = (
            historical["interaction_type"]
            .eq("SUPPORT_REQUEST")
            .sum()
        )

        negative_count = (
            historical["sentiment"]
            .eq("NEGATIVE")
            .sum()
        )

        escalation_count = (
            historical["interaction_type"]
            .eq("ESCALATION")
            .sum()
        )

        recent = historical[
            historical["interaction_date"]
            >= recent_start
        ]

        recent_complaint_count = (
            recent["interaction_type"]
            .eq("COMPLAINT")
            .sum()
        )

        recent_negative_count = (
            recent["sentiment"]
            .eq("NEGATIVE")
            .sum()
        )

        return {
            "interaction_count": int(
                len(historical)
            ),
            "complaint_count": int(
                complaint_count
            ),
            "support_interaction_count": int(
                support_count
            ),
            "negative_sentiment_count": int(
                negative_count
            ),
            "escalation_count": int(
                escalation_count
            ),
            "recent_complaint_count": int(
                recent_complaint_count
            ),
            "recent_negative_sentiment_count": int(
                recent_negative_count
            ),
        }

    # ==========================================================
    # DOCUMENT FEATURES
    # ==========================================================

    def _document_features(
        self,
        customer_id: str,
        observation_date: pd.Timestamp,
    ) -> dict:

        documents = self.document_groups.get(
            customer_id,
            pd.DataFrame(),
        )

        if documents.empty:
            return {
                "document_count": 0,
                "complaint_document_count": 0,
                "support_document_count": 0,
            }

        lookback_start = (
            observation_date
            - pd.Timedelta(
                days=settings.FEATURE_LOOKBACK_DAYS - 1
            )
        )

        historical = documents[
            (documents["document_date"] >= lookback_start)
            & (
                documents["document_date"]
                <= observation_date
            )
        ].copy()

        if historical.empty:
            return {
                "document_count": 0,
                "complaint_document_count": 0,
                "support_document_count": 0,
            }

        complaint_documents = (
            historical["document_type"]
            .eq("COMPLAINT")
            .sum()
        )

        support_documents = (
            historical["document_type"]
            .eq("SUPPORT_NOTE")
            .sum()
        )

        return {
            "document_count": int(
                len(historical)
            ),
            "complaint_document_count": int(
                complaint_documents
            ),
            "support_document_count": int(
                support_documents
            ),
        }

    # ==========================================================
    # FUTURE TARGET
    # ==========================================================

    def _build_target(
        self,
        customer_id: str,
        observation_date: pd.Timestamp,
        previous_transaction_count: int,
    ) -> tuple[int, dict]:

        target_end = (
            observation_date
            + pd.Timedelta(
                days=settings.PREDICTION_WINDOW_DAYS
            )
        )

        transactions = self.transaction_groups.get(
            customer_id,
            pd.DataFrame(),
        )

        interactions = self.interaction_groups.get(
            customer_id,
            pd.DataFrame(),
        )

        # ------------------------------------------------------
        # Future transactions
        # ------------------------------------------------------

        if transactions.empty:
            future_transactions = pd.DataFrame()
        else:
            future_transactions = transactions[
                (transactions["transaction_date"] > observation_date)
                & (
                    transactions["transaction_date"]
                    <= target_end
                )
            ].copy()

        future_transaction_count = len(
            future_transactions
        )

        if future_transactions.empty:
            future_failed_count = 0
            future_overdue_count = 0
            future_overdue_amount = 0.0
        else:
            future_failed_count = (
                future_transactions["payment_status"]
                .eq("FAILED")
                .sum()
            )

            # --------------------------------------------------
            # Was the transaction overdue by the end of the
            # 30-day prediction window?
            # --------------------------------------------------

            future_overdue_mask = (
                (future_transactions["due_date"] <= target_end)
                & (
                    future_transactions["payment_date"].isna()
                    | (
                        future_transactions["payment_date"]
                        > target_end
                    )
                )
            )

            future_overdue = future_transactions[
                future_overdue_mask
            ]

            future_overdue_count = len(
                future_overdue
            )

            future_overdue_amount = (
                future_overdue[
                    "transaction_amount"
                ].sum()
                if not future_overdue.empty
                else 0.0
            )

        # ------------------------------------------------------
        # Future interactions
        # ------------------------------------------------------

        if interactions.empty:
            future_interactions = pd.DataFrame()
        else:
            future_interactions = interactions[
                (interactions["interaction_date"] > observation_date)
                & (
                    interactions["interaction_date"]
                    <= target_end
                )
            ].copy()

        if future_interactions.empty:
            future_complaint_count = 0
            future_negative_count = 0
            future_escalation_count = 0
        else:
            future_complaint_count = (
                future_interactions["interaction_type"]
                .eq("COMPLAINT")
                .sum()
            )

            future_negative_count = (
                future_interactions["sentiment"]
                .eq("NEGATIVE")
                .sum()
            )

            future_escalation_count = (
                future_interactions["interaction_type"]
                .eq("ESCALATION")
                .sum()
            )

        # ------------------------------------------------------
        # Future activity decline
        # ------------------------------------------------------

        if previous_transaction_count >= 2:
            activity_decline_flag = int(
                future_transaction_count
                < (
                    previous_transaction_count
                    * 0.50
                )
            )
        else:
            activity_decline_flag = 0

        # ------------------------------------------------------
        # Risk indicators
        # ------------------------------------------------------

        overdue_flag = int(
            future_overdue_count >= 1
        )

        failed_flag = int(
            future_failed_count >= 1
        )

        complaint_flag = int(
            future_complaint_count >= 1
        )

        negative_flag = int(
            future_negative_count >= 1
        )

        escalation_flag = int(
            future_escalation_count >= 1
        )

        # ------------------------------------------------------
        # Business risk score
        #
        # This is our documented target methodology.
        #
        # 30% -> overdue behavior
        # 25% -> failed payments
        # 15% -> complaints
        # 10% -> negative sentiment
        # 10% -> escalations
        # 10% -> declining activity
        # ------------------------------------------------------

        risk_score = (
            (0.30 * overdue_flag)
            + (0.25 * failed_flag)
            + (0.15 * complaint_flag)
            + (0.10 * negative_flag)
            + (0.10 * escalation_flag)
            + (0.10 * activity_decline_flag)
        )

        target = int(
            risk_score
            >= settings.TARGET_RISK_SCORE_THRESHOLD
        )

        audit = {
            "future_transaction_count": int(
                future_transaction_count
            ),
            "future_failed_count": int(
                future_failed_count
            ),
            "future_overdue_count": int(
                future_overdue_count
            ),
            "future_overdue_amount": float(
                future_overdue_amount
            ),
            "future_complaint_count": int(
                future_complaint_count
            ),
            "future_negative_count": int(
                future_negative_count
            ),
            "future_escalation_count": int(
                future_escalation_count
            ),
            "activity_decline_flag": int(
                activity_decline_flag
            ),
            "risk_score": float(
                risk_score
            ),
            "target": target,
        }

        return target, audit

    # ==========================================================
    # CUSTOMER SNAPSHOT
    # ==========================================================

    def _build_snapshot(
        self,
        customer: pd.Series,
        observation_date: pd.Timestamp,
    ) -> tuple[dict, dict]:

        customer_id = customer["customer_id"]

        # ------------------------------------------------------
        # Customer must already exist on observation date.
        # ------------------------------------------------------

        customer_since = pd.Timestamp(
            customer["customer_since"]
        )

        if customer_since > observation_date:
            return None, None

        # ------------------------------------------------------
        # Customer features
        # ------------------------------------------------------

        tenure_days = (
            observation_date - customer_since
        ).days

        feature_row = {
            "customer_id": customer_id,
            "observation_date": observation_date.date(),
            "customer_segment": customer["customer_segment"],
            "industry": customer["industry"],
            "country": customer["country"],
            "account_status": customer["account_status"],
            "customer_tenure_days": int(
                max(tenure_days, 0)
            ),
        }

        # ------------------------------------------------------
        # Transaction features
        # ------------------------------------------------------

        transaction_features = (
            self._transaction_features(
                customer_id,
                observation_date,
            )
        )

        feature_row.update(
            transaction_features
        )

        # ------------------------------------------------------
        # Interaction features
        # ------------------------------------------------------

        interaction_features = (
            self._interaction_features(
                customer_id,
                observation_date,
            )
        )

        feature_row.update(
            interaction_features
        )

        # ------------------------------------------------------
        # Document features
        # ------------------------------------------------------

        document_features = (
            self._document_features(
                customer_id,
                observation_date,
            )
        )

        feature_row.update(
            document_features
        )

        # ------------------------------------------------------
        # Future target
        # ------------------------------------------------------

        target, audit = self._build_target(
            customer_id,
            observation_date,
            transaction_features[
                "previous_transaction_count"
            ],
        )

        feature_row["target"] = target

        # Add values useful for audit/debugging.
        audit_row = {
            "customer_id": customer_id,
            "observation_date": observation_date.date(),
            **audit,
        }

        return feature_row, audit_row

    # ==========================================================
    # BUILD COMPLETE DATASET
    # ==========================================================

    def build(self) -> tuple[pd.DataFrame, pd.DataFrame]:

        observation_dates = (
            self.get_observation_dates()
        )

        feature_rows = []
        audit_rows = []

        print("\n" + "=" * 70)
        print("FEATURE ENGINEERING")
        print("=" * 70)

        print(
            f"\nFeature lookback: "
            f"{settings.FEATURE_LOOKBACK_DAYS} days"
        )

        print(
            f"Prediction window: "
            f"{settings.PREDICTION_WINDOW_DAYS} days"
        )

        print(
            f"Target risk threshold: "
            f"{settings.TARGET_RISK_SCORE_THRESHOLD}"
        )

        print(
            f"Observation dates: "
            f"{len(observation_dates)}"
        )

        # ------------------------------------------------------
        # Build one snapshot per customer per observation date.
        # ------------------------------------------------------

        for index, observation_date in enumerate(
            observation_dates,
            start=1,
        ):

            snapshot_count = 0

            for _, customer in self.customers.iterrows():

                feature_row, audit_row = (
                    self._build_snapshot(
                        customer,
                        observation_date,
                    )
                )

                if feature_row is None:
                    continue

                feature_rows.append(
                    feature_row
                )

                audit_rows.append(
                    audit_row
                )

                snapshot_count += 1

            print(
                f"[{index}/{len(observation_dates)}] "
                f"{observation_date.date()} "
                f"-> {snapshot_count:,} customer snapshots"
            )

        features_df = pd.DataFrame(
            feature_rows
        )

        audit_df = pd.DataFrame(
            audit_rows
        )

        return features_df, audit_df

    # ==========================================================
    # SAVE OUTPUT
    # ==========================================================

    def save(
        self,
        features_df: pd.DataFrame,
        audit_df: pd.DataFrame,
    ) -> None:

        ML_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        feature_path = (
            ML_DIR / "customer_features.csv"
        )

        audit_path = (
            ML_DIR / "target_audit.csv"
        )

        features_df.to_csv(
            feature_path,
            index=False,
        )

        audit_df.to_csv(
            audit_path,
            index=False,
        )

        print("\n" + "=" * 70)
        print("FEATURE DATASET SAVED")
        print("=" * 70)

        print(
            f"\nFeature dataset: {feature_path}"
        )

        print(
            f"Target audit:    {audit_path}"
        )

        print(
            f"\nFeature rows: "
            f"{len(features_df):,}"
        )

        print(
            f"Feature columns: "
            f"{len(features_df.columns):,}"
        )

        # ------------------------------------------------------
        # Target distribution
        # ------------------------------------------------------

        target_counts = (
            features_df["target"]
            .value_counts()
            .sort_index()
        )

        print("\nTarget distribution:")

        for target_value, count in (
            target_counts.items()
        ):
            percentage = (
                count
                / len(features_df)
                * 100
            )

            label = (
                "HIGH RISK"
                if target_value == 1
                else "NOT HIGH RISK"
            )

            print(
                f"  {target_value} ({label}): "
                f"{count:,} "
                f"({percentage:.2f}%)"
            )

    # ==========================================================
    # RUN
    # ==========================================================

    def run(self) -> None:

        features_df, audit_df = self.build()

        self.save(
            features_df,
            audit_df,
        )


# ==============================================================
# MAIN
# ==============================================================

def main() -> None:

    print("\n" + "=" * 70)
    print("CUSTOMER RISK INTELLIGENCE")
    print("CUSTOMER FEATURE ENGINEERING PIPELINE")
    print("=" * 70)

    engineer = CustomerFeatureEngineer()

    engineer.run()

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
