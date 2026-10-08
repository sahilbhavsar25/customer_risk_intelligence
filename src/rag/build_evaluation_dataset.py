"""
Build the RAG evaluation set.

    python -m src.rag.build_evaluation_dataset

Case types:

    document         answer must come from a specific document
    structured_data  answer must come from profile / history /
                     model output (exact facts and numbers)
    unavailable      information genuinely does not exist for
                     the customer -> exact fallback expected

Unavailable cases are verified against the data before they
are saved (no document of that type, nothing in structured
data that answers it, retriever returns nothing). They are NOT
filtered on what the assistant answers - that is what the
evaluation measures, and filtering on it would inflate the
metrics.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.rag.retriever import (
    CustomerDocumentRetriever,
    plan_query,
)


# ==============================================================
# PATHS
# ==============================================================

CUSTOMERS_PATH = Path(
    "data/processed/customers/customers_clean.csv"
)

DOCUMENTS_PATH = Path(
    "data/processed/documents/documents_clean.csv"
)

TRANSACTIONS_PATH = Path(
    "data/processed/transactions/transactions_clean.csv"
)

INTERACTIONS_PATH = Path(
    "data/processed/interactions/interactions_clean.csv"
)

PREDICTIONS_PATH = Path(
    "data/processed/ml/explainability/"
    "customer_risk_predictions.csv"
)

OUTPUT_PATH = Path(
    "data/evaluation/rag_evaluation.json"
)

RANDOM_SEED = 42

FALLBACK_MESSAGE = (
    "I could not find sufficient information in the "
    "available customer data."
)


# ==============================================================
# DOCUMENT TYPES / QUESTIONS
# ==============================================================

DOCUMENT_TYPES = [
    "PAYMENT_ISSUE",
    "COMPLAINT",
    "SUPPORT_NOTE",
    "RENEWAL_DISCUSSION",
    "ESCALATION",
    "ACCOUNT_NOTE",
    "CUSTOMER_PREFERENCE",
]

# Second question per type, so the three extra cases don't all
# reuse the same wording.
QUESTION_TEMPLATES = {
    "PAYMENT_ISSUE": [
        "What payment issue was reported for this customer?",
        "What payment problems has this customer experienced?",
    ],
    "COMPLAINT": [
        "What complaint did this customer raise?",
        "Has this customer complained about anything?",
    ],
    "SUPPORT_NOTE": [
        "What support issue did this customer report?",
        "What did the support team record for this customer?",
    ],
    "RENEWAL_DISCUSSION": [
        "What renewal discussion was recorded for this customer?",
    ],
    "ESCALATION": [
        "What escalation was recorded for this customer?",
    ],
    "ACCOUNT_NOTE": [
        "What account-related note was recorded for this "
        "customer?",
    ],
    "CUSTOMER_PREFERENCE": [
        "What customer preference was recorded?",
    ],
}


# ==============================================================
# KEY FACTS PER DOCUMENT TEMPLATE
#
# Each generated document comes from a fixed template. These
# are the facts a correct answer has to mention.
# ==============================================================

TEMPLATE_FACTS = [
    (
        "repeated service issues",
        [
            "repeated service issues",
            "business operations",
            "quick resolution",
        ],
    ),
    (
        "complaint about delayed service response",
        [
            "delayed service response",
            "follow-up",
        ],
    ),
    (
        "dissatisfaction with the recent service experience",
        [
            "dissatisfaction",
            "service experience",
            "escalated",
        ],
    ),
    (
        "reported a payment-related issue",
        [
            "payment delayed",
            "outstanding amount",
            "clarification",
        ],
    ),
    (
        "additional time to resolve an outstanding payment",
        [
            "additional time",
            "outstanding payment",
            "payment status reviewed",
        ],
    ),
    (
        "regarding an overdue invoice",
        [
            "overdue invoice",
            "due amount",
            "payment timeline",
        ],
    ),
    (
        "assistance with a technical issue",
        [
            "technical issue",
            "troubleshooting guidance",
        ],
    ),
    (
        "with an account-related issue",
        [
            "account-related issue",
            "guidance",
        ],
    ),
    (
        "contacted the support team for assistance",
        [
            "assistance",
            "follow-up",
            "monitoring",
        ],
    ),
    (
        "information about renewal terms",
        [
            "renewal terms",
            "pricing",
            "options",
        ],
    ),
    (
        "discussed the upcoming renewal",
        [
            "upcoming renewal",
            "contract terms",
            "service continuity",
        ],
    ),
    (
        "escalated to a senior support team",
        [
            "escalated",
            "senior support team",
            "additional investigation",
        ],
    ),
    (
        "requested escalation after the initial support",
        [
            "escalation",
            "initial support interaction",
            "not fully resolve",
        ],
    ),
    (
        "account review completed",
        [
            "account review",
            "account management",
        ],
    ),
    (
        "recent account activity were reviewed",
        [
            "customer profile",
            "account activity",
            "account management team",
        ],
    ),
    (
        "prefers communication through email",
        [
            "email",
            "account updates",
        ],
    ),
    (
        "support updates through the online portal",
        [
            "online portal",
            "support updates",
        ],
    ),
]


def facts_for_content(content: str) -> list[str]:

    content_lower = content.lower()

    for marker, facts in TEMPLATE_FACTS:
        if marker in content_lower:
            return facts

    raise ValueError(
        f"No key facts defined for document content: "
        f"{content[:80]}"
    )


# ==============================================================
# LOAD DATA
# ==============================================================

def load_data() -> dict[str, pd.DataFrame]:

    paths = {
        "customers": CUSTOMERS_PATH,
        "documents": DOCUMENTS_PATH,
        "transactions": TRANSACTIONS_PATH,
        "interactions": INTERACTIONS_PATH,
        "predictions": PREDICTIONS_PATH,
    }

    data = {}

    for name, path in paths.items():

        if not path.exists():
            raise FileNotFoundError(
                f"{name} dataset not found: {path}"
            )

        data[name] = pd.read_csv(path)

    return data


# ==============================================================
# DOCUMENT-BACKED QUESTIONS
# ==============================================================

def create_document_cases(
    data: dict[str, pd.DataFrame],
    rng: np.random.Generator,
) -> list[dict]:

    documents = data["documents"]

    predicted_customers = set(
        data["predictions"]["customer_id"]
    )

    # ----------------------------------------------------------
    # Only use (customer, type) pairs with exactly one document
    # of that type, so the expected source is unambiguous.
    # ----------------------------------------------------------

    type_counts = (
        documents
        .groupby(["customer_id", "document_type"])
        .size()
        .rename("type_count")
        .reset_index()
    )

    candidates = documents.merge(
        type_counts,
        on=["customer_id", "document_type"],
    )

    candidates = candidates[
        (candidates["type_count"] == 1)
        & candidates["document_type"].isin(DOCUMENT_TYPES)
    ]

    selected = []
    used_customers = set()

    plan = (
        [(document_type, 0) for document_type in DOCUMENT_TYPES]
        + [
            ("PAYMENT_ISSUE", 1),
            ("COMPLAINT", 1),
            ("SUPPORT_NOTE", 1),
        ]
    )

    for position, (document_type, template_index) in enumerate(plan):

        pool = candidates[
            (candidates["document_type"] == document_type)
            & ~candidates["customer_id"].isin(used_customers)
        ]

        # ------------------------------------------------------
        # Alternate between customers with and without an ML
        # snapshot - the second group used to be reported as
        # "Customer not found".
        # ------------------------------------------------------

        want_snapshot = position % 2 == 0

        preferred = pool[
            pool["customer_id"].isin(predicted_customers)
            == want_snapshot
        ]

        if not preferred.empty:
            pool = preferred

        if pool.empty:
            raise ValueError(
                f"No unambiguous document left for type "
                f"{document_type}"
            )

        row = pool.iloc[
            int(rng.integers(0, len(pool)))
        ]

        used_customers.add(row["customer_id"])

        selected.append((row, template_index))

    cases = []

    for row, template_index in selected:

        document_type = row["document_type"]

        cases.append(
            {
                "customer_id": str(row["customer_id"]),
                "question": QUESTION_TEMPLATES[
                    document_type
                ][template_index],
                "question_type": "document",
                "intentionally_unavailable": False,
                "has_ml_prediction": (
                    row["customer_id"] in predicted_customers
                ),
                "expected_answer": str(row["content"]),
                "expected_sources": [
                    str(row["document_id"])
                ],
                "expected_source_metadata": {
                    "document_id": str(row["document_id"]),
                    "document_type": str(document_type),
                    "document_date": str(row["document_date"]),
                    "source": str(row["source"]),
                },
                "expected_facts": [
                    {
                        "type": "phrase",
                        "value": fact,
                    }
                    for fact in facts_for_content(
                        str(row["content"])
                    )
                ],
            }
        )

    return cases


# ==============================================================
# STRUCTURED-DATA QUESTIONS
# ==============================================================

def create_structured_cases(
    data: dict[str, pd.DataFrame],
    rng: np.random.Generator,
) -> list[dict]:

    customers = data["customers"]
    transactions = data["transactions"]
    predictions = data["predictions"]

    cases = []

    # ----------------------------------------------------------
    # 1. Failed + overdue counts. Pick a customer where both are
    #    non-zero and different, so the numbers can't be mixed
    #    up and still pass.
    # ----------------------------------------------------------

    status_counts = (
        transactions
        .pivot_table(
            index="customer_id",
            columns="payment_status",
            values="transaction_id",
            aggfunc="count",
            fill_value=0,
        )
    )

    eligible = status_counts[
        (status_counts["FAILED"] > 0)
        & (status_counts["OVERDUE"] > 0)
        & (status_counts["FAILED"] != status_counts["OVERDUE"])
    ]

    customer_id = eligible.index[
        int(rng.integers(0, len(eligible)))
    ]

    failed = int(eligible.loc[customer_id, "FAILED"])
    overdue = int(eligible.loc[customer_id, "OVERDUE"])

    cases.append(
        {
            "customer_id": customer_id,
            "question": (
                "How many failed transactions and how many "
                "overdue transactions does this customer have "
                "in total?"
            ),
            "question_type": "structured_data",
            "intentionally_unavailable": False,
            "expected_answer": (
                f"failed transactions = {failed}, "
                f"overdue transactions = {overdue}"
            ),
            "expected_sources": [],
            "expected_facts": [
                {"type": "count", "label": "failed", "value": failed},
                {"type": "count", "label": "overdue", "value": overdue},
            ],
        }
    )

    # ----------------------------------------------------------
    # 2. Profile facts from customer master.
    # ----------------------------------------------------------

    known = customers[
        (customers["industry"] != "Unknown")
        & (customers["account_status"] != "UNKNOWN")
    ]

    row = known.iloc[int(rng.integers(0, len(known)))]

    cases.append(
        {
            "customer_id": str(row["customer_id"]),
            "question": (
                "Which segment and industry does this customer "
                "belong to, and what is the account status?"
            ),
            "question_type": "structured_data",
            "intentionally_unavailable": False,
            "expected_answer": (
                f"segment = {row['customer_segment']}, "
                f"industry = {row['industry']}, "
                f"account status = {row['account_status']}"
            ),
            "expected_sources": [],
            "expected_facts": [
                {"type": "text", "value": str(row["customer_segment"])},
                {"type": "text", "value": str(row["industry"])},
                {"type": "text", "value": str(row["account_status"])},
            ],
        }
    )

    # ----------------------------------------------------------
    # 3. Model output for a customer that has a prediction.
    # ----------------------------------------------------------

    latest_predictions = (
        predictions
        .sort_values("observation_date")
        .groupby("customer_id")
        .tail(1)
    )

    row = latest_predictions.iloc[
        int(rng.integers(0, len(latest_predictions)))
    ]

    probability = round(
        float(row["predicted_risk_probability"]),
        4,
    )

    cases.append(
        {
            "customer_id": str(row["customer_id"]),
            "question": (
                "What is this customer's current ML risk level "
                "and risk probability?"
            ),
            "question_type": "structured_data",
            "intentionally_unavailable": False,
            "expected_answer": (
                f"risk level = {row['risk_level']}, "
                f"risk probability = {probability}"
            ),
            "expected_sources": [],
            "expected_facts": [
                {"type": "text", "value": str(row["risk_level"])},
                {"type": "probability", "value": probability},
            ],
        }
    )

    # ----------------------------------------------------------
    # 4. Same question for a valid customer WITHOUT a model
    #    snapshot. Correct behaviour: say it is unavailable and
    #    do not invent a probability.
    # ----------------------------------------------------------

    no_prediction = customers[
        ~customers["customer_id"].isin(
            predictions["customer_id"]
        )
    ]

    row = no_prediction.iloc[
        int(rng.integers(0, len(no_prediction)))
    ]

    cases.append(
        {
            "customer_id": str(row["customer_id"]),
            "question": (
                "What is this customer's current ML risk level "
                "and risk probability?"
            ),
            "question_type": "structured_data",
            "intentionally_unavailable": False,
            "has_ml_prediction": False,
            "expected_answer": (
                "ML risk prediction is unavailable because this "
                "customer does not have sufficient historical "
                "data for the current prediction framework."
            ),
            "expected_sources": [],
            "expected_facts": [
                {"type": "phrase", "value": "unavailable"},
                {
                    "type": "absent_pattern",
                    "value": (
                        r"\b0\.\d+|\d+(\.\d+)?\s?%"
                        r"|\b(high|medium|low)\s+risk\b"
                    ),
                },
            ],
        }
    )

    return cases


# ==============================================================
# INTENTIONALLY UNAVAILABLE QUESTIONS
# ==============================================================

def create_unavailable_cases(
    data: dict[str, pd.DataFrame],
    retriever: CustomerDocumentRetriever | None,
) -> list[dict]:

    documents = data["documents"]
    interactions = data["interactions"]

    # Customers that DO have documents, so the customer filter
    # alone doesn't make these trivially empty.
    customers_with_documents = sorted(
        documents["customer_id"].unique()
    )

    def customers_with_type(document_type: str) -> set:
        return set(
            documents.loc[
                documents["document_type"] == document_type,
                "customer_id",
            ]
        )

    renewal_interaction_customers = set(
        interactions.loc[
            interactions["interaction_type"] == "RENEWAL",
            "customer_id",
        ]
    )

    specs = [
        {
            "question": "What customer preference was recorded?",
            # There is no preference field in structured data.
            "candidates": [
                customer
                for customer in customers_with_documents
                if customer
                not in customers_with_type("CUSTOMER_PREFERENCE")
            ],
            "reason": (
                "Customer has documents but no "
                "CUSTOMER_PREFERENCE document; structured data "
                "has no preference field."
            ),
        },
        {
            "question": (
                "What renewal discussion was recorded for this "
                "customer?"
            ),
            # Must also have no RENEWAL interaction, otherwise
            # structured data could legitimately answer it.
            "candidates": [
                customer
                for customer in customers_with_documents
                if customer
                not in customers_with_type("RENEWAL_DISCUSSION")
                and customer not in renewal_interaction_customers
            ],
            "reason": (
                "No RENEWAL_DISCUSSION document and no RENEWAL "
                "interaction for this customer."
            ),
        },
        {
            "question": "What legal dispute did this customer report?",
            "candidates": [
                customer
                for customer in customers_with_documents
                if not documents.loc[
                    documents["customer_id"] == customer,
                    "content",
                ]
                .str.contains(
                    "legal|lawsuit|dispute|court",
                    case=False,
                )
                .any()
            ],
            "reason": (
                "Legal disputes are not a supported document "
                "type and no document mentions one."
            ),
        },
    ]

    cases = []
    used = set()

    for spec in specs:

        candidates = [
            customer
            for customer in spec["candidates"]
            if customer not in used
        ]

        if not candidates:
            raise ValueError(
                f"No genuinely unavailable customer for: "
                f"{spec['question']}"
            )

        # Middle of the sorted list rather than the first ID,
        # just to avoid always testing CUST_0000x.
        customer_id = candidates[len(candidates) // 2]
        used.add(customer_id)

        # ------------------------------------------------------
        # Retriever check: must return nothing.
        # ------------------------------------------------------

        retrieved = []

        if retriever is not None:
            retrieved = retriever.retrieve(
                customer_id,
                spec["question"],
            )

        if retrieved:
            raise ValueError(
                f"Unavailable case {customer_id} / "
                f"{spec['question']!r} retrieved documents "
                f"{[r['document_id'] for r in retrieved]}. "
                f"Either the case is not genuinely unavailable "
                f"or retrieval filtering is broken."
            )

        cases.append(
            {
                "customer_id": customer_id,
                "question": spec["question"],
                "question_type": "unavailable",
                "intentionally_unavailable": True,
                "expected_answer": FALLBACK_MESSAGE,
                "expected_sources": [],
                "expected_facts": [],
                "verification": {
                    "reason": spec["reason"],
                    "query_plan": plan_query(
                        spec["question"]
                    ).mode,
                    "retriever_checked": retriever is not None,
                    "retrieved_documents": 0,
                },
            }
        )

    return cases


# ==============================================================
# VALIDATE / SAVE
# ==============================================================

def validate_cases(cases: list[dict]) -> None:

    counts = pd.Series(
        [case["question_type"] for case in cases]
    ).value_counts()

    if len(cases) < 15:
        raise ValueError(
            f"Only {len(cases)} cases; minimum is 15."
        )

    if counts.get("document", 0) < 10:
        raise ValueError("Need at least 10 document cases.")

    if counts.get("structured_data", 0) < 2:
        raise ValueError("Need at least 2 structured cases.")

    if counts.get("unavailable", 0) < 3:
        raise ValueError("Need at least 3 unavailable cases.")


def save_cases(cases: list[dict]) -> None:

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            cases,
            file,
            indent=4,
            ensure_ascii=False,
            default=lambda value: (
                bool(value)
                if isinstance(value, np.bool_)
                else int(value)
                if isinstance(value, np.integer)
                else str(value)
            ),
        )

    print(f"\nSaved evaluation dataset: {OUTPUT_PATH}")


def print_summary(cases: list[dict]) -> None:

    print("\n" + "=" * 70)
    print("RAG EVALUATION DATASET")
    print("=" * 70)

    counts = pd.Series(
        [case["question_type"] for case in cases]
    ).value_counts()

    print(f"Total questions:       {len(cases)}")

    for question_type, count in counts.items():
        print(f"{question_type + ':':<22} {count}")

    print("\nQuestions:")

    for case in cases:
        print(
            f"{case['eval_id']} | "
            f"{case['customer_id']} | "
            f"{case['question_type']:<15} | "
            f"{case['question']}"
        )


# ==============================================================
# MAIN
# ==============================================================

def main():

    print("\n" + "=" * 70)
    print("CUSTOMER RISK INTELLIGENCE")
    print("BUILD RAG EVALUATION DATASET")
    print("=" * 70)

    data = load_data()

    rng = np.random.default_rng(RANDOM_SEED)

    try:
        retriever = CustomerDocumentRetriever()
    except Exception as exc:
        # Data-level checks still run; only the retriever
        # sanity check is skipped.
        print(
            f"[WARN] Retriever unavailable ({exc}); "
            f"skipping retrieval check for unavailable cases."
        )
        retriever = None

    cases = (
        create_document_cases(data, rng)
        + create_structured_cases(data, rng)
        + create_unavailable_cases(data, retriever)
    )

    for index, case in enumerate(cases, start=1):
        case["eval_id"] = f"RAG_{index:03d}"

    # eval_id first for readability.
    cases = [
        {"eval_id": case.pop("eval_id"), **case}
        for case in cases
    ]

    validate_cases(cases)
    save_cases(cases)
    print_summary(cases)


if __name__ == "__main__":
    main()
