"""
Evaluate the customer intelligence assistant.

    python -m src.rag.evaluate_rag

Metrics are scoped by case type:

                      retrieval  answer  citation  grounding
    document              x        x        x          x
    structured_data       -        x        -          x
    unavailable           x        x        x          x

"-" means not applicable (stored as null and excluded from the
metric), instead of being counted as a failure.
"""

from pathlib import Path
import json
import math
import re
import time

import pandas as pd

from src.rag.customer_intelligence import (
    FALLBACK_MESSAGE,
    CustomerIntelligenceService,
)


# ==============================================================
# PATHS
# ==============================================================

EVALUATION_PATH = Path(
    "data/evaluation/rag_evaluation.json"
)

DOCUMENTS_PATH = Path(
    "data/processed/documents/documents_clean.csv"
)

OUTPUT_PATH = Path(
    "data/evaluation/rag_evaluation_results.json"
)

CASES_CSV_PATH = Path(
    "data/evaluation/rag_evaluation_cases.csv"
)

SUMMARY_PATH = Path(
    "data/evaluation/rag_evaluation_summary.csv"
)


# ==============================================================
# THRESHOLDS
#
# Fixed up-front, not tuned to make results look better.
# ==============================================================

# Share of a document's key facts the answer must contain.
MIN_FACT_COVERAGE = 0.6

# Share of an expected phrase's words that must appear.
MIN_PHRASE_WORD_MATCH = 2 / 3

# Share of the answer's content words that must be found in
# the evidence given to the model.
MIN_EVIDENCE_SUPPORT = 0.7

# Share of the answer's content words that must be found in
# a cited document for that citation to count as supporting.
MIN_CITATION_SUPPORT = 0.5

STEM_LENGTH = 5

STOP_WORDS = {
    "the", "a", "an", "is", "was", "were", "be", "been", "has",
    "have", "had", "this", "that", "these", "those", "for",
    "with", "and", "or", "of", "to", "in", "on", "at", "by",
    "from", "as", "it", "its", "their", "they", "them", "there",
    "which", "who", "what", "when", "also", "any", "about",
    "into", "than", "then", "but", "not", "no", "are", "can",
    "did", "does", "do", "per", "via", "based", "according",
    "customer", "customers", "recorded", "record", "records",
    "data", "information", "structured", "document", "documents",
    "noted", "note", "indicates", "indicated", "shows", "show",
    "mentioned", "mentions", "regarding", "related", "specific",
    "currently", "current", "total", "following", "including",
    "includes", "provided", "available", "reported", "report",
    "described", "states", "stated", "details", "detail",
    "experienced", "involving", "involved", "specifically",
}

NEGATION_PATTERNS = [
    r"\bno record",
    r"\bnot recorded\b",
    r"\bno information\b",
    r"\bcould not find\b",
    r"\bdoes not have any\b",
    r"\bthere (is|are) no\b",
]


# ==============================================================
# TEXT HELPERS
# ==============================================================

def normalize_text(text: str) -> str:

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def stems(text: str) -> set[str]:

    return {
        word[:STEM_LENGTH]
        for word in normalize_text(text).split()
        if word not in STOP_WORDS
        and not word.isdigit()
        and len(word) > 2
    }


def content_stems(text: str) -> list[str]:

    return [
        word[:STEM_LENGTH]
        for word in normalize_text(text).split()
        if word not in STOP_WORDS
        and not word.isdigit()
        and len(word) > 2
    ]


def phrase_present(
    phrase: str,
    answer: str,
) -> bool:

    phrase_stems = stems(phrase)

    if not phrase_stems:
        return False

    found = phrase_stems & stems(answer)

    return (
        len(found) / len(phrase_stems)
        >= MIN_PHRASE_WORD_MATCH
    )


def extract_numbers(text: str) -> list[tuple[float, bool]]:
    """
    Returns (value, is_percentage) pairs.
    """

    numbers = []

    for match in re.finditer(
        r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(\s?%)?",
        str(text),
    ):
        value = float(match.group(1).replace(",", ""))
        numbers.append((value, bool(match.group(2))))

    return numbers


def number_supported(
    value: float,
    is_percentage: bool,
    evidence_numbers: list[float],
) -> bool:

    for evidence in evidence_numbers:

        if math.isclose(value, evidence, abs_tol=0.006):
            return True

        if is_percentage and math.isclose(
            value / 100,
            evidence,
            abs_tol=0.0006,
        ):
            return True

    return False


def support_ratio(
    answer: str,
    evidence: str,
) -> float:

    answer_words = content_stems(answer)

    if not answer_words:
        return 1.0

    evidence_words = stems(evidence)

    return sum(
        word in evidence_words
        for word in answer_words
    ) / len(answer_words)


# ==============================================================
# FACT CHECKS (answer correctness)
# ==============================================================

def count_mentioned(
    label: str,
    value: int,
    answer: str,
) -> bool:
    """
    "3 failed transactions", "failed transactions: 3",
    "Failed = 3" ... The number has to sit next to the label,
    so swapping the two counts fails.
    """

    text = answer.lower()
    number = rf"\b{value}\b"
    gap = r"(?:\W+\w+){0,3}?\W+"

    return bool(
        re.search(number + gap + label, text)
        or re.search(label + gap + number, text)
        or re.search(number + r"\W+" + label, text)
    )


def probability_mentioned(
    value: float,
    answer: str,
) -> bool:

    return any(
        number_supported(number, is_percentage, [value])
        for number, is_percentage in extract_numbers(answer)
    )


def check_fact(
    fact: dict,
    answer: str,
) -> bool:

    fact_type = fact["type"]

    if fact_type == "phrase":
        return phrase_present(fact["value"], answer)

    if fact_type == "text":
        return (
            normalize_text(fact["value"])
            in normalize_text(answer)
        )

    if fact_type == "count":
        return count_mentioned(
            fact["label"],
            int(fact["value"]),
            answer,
        )

    if fact_type == "probability":
        return probability_mentioned(
            float(fact["value"]),
            answer,
        )

    if fact_type == "absent_pattern":
        return not re.search(
            fact["value"],
            answer,
            flags=re.IGNORECASE,
        )

    raise ValueError(f"Unknown fact type: {fact_type}")


def evaluate_answer(
    case: dict,
    result: dict,
) -> tuple[bool, dict]:

    answer = result.get("answer", "") or ""

    if case["question_type"] == "unavailable":
        return answer.strip() == FALLBACK_MESSAGE, {}

    if answer.strip() == FALLBACK_MESSAGE:
        return False, {"reason": "fallback_returned"}

    facts = case.get("expected_facts", [])

    fact_results = {
        f"{fact['type']}:{fact.get('label', fact['value'])}": (
            check_fact(fact, answer)
        )
        for fact in facts
    }

    if case["question_type"] == "document":

        # Answer must not contradict the source ("no record of
        # a complaint" when the document is a complaint).
        contradiction = any(
            re.search(pattern, answer.lower())
            for pattern in NEGATION_PATTERNS
        )

        coverage = (
            sum(fact_results.values()) / len(fact_results)
            if fact_results
            else 0.0
        )

        return (
            coverage >= MIN_FACT_COVERAGE
            and not contradiction
        ), {
            "facts": fact_results,
            "fact_coverage": round(coverage, 3),
            "contradiction": contradiction,
        }

    # Structured: every expected fact / number must be right.
    return all(fact_results.values()), {
        "facts": fact_results,
    }


# ==============================================================
# RETRIEVAL
# ==============================================================

def evaluate_retrieval(
    case: dict,
    result: dict,
) -> bool | None:

    retrieved = set(
        result.get("retrieved_document_ids", [])
    )

    if case["question_type"] == "structured_data":
        return None

    if case["question_type"] == "unavailable":
        return not retrieved

    return bool(
        set(case["expected_sources"]) & retrieved
    )


# ==============================================================
# CITATIONS
# ==============================================================

def evaluate_citations(
    case: dict,
    result: dict,
    documents: pd.DataFrame,
) -> tuple[bool | None, dict]:

    cited = [
        source.get("document_id")
        for source in result.get("sources", [])
    ]

    if case["question_type"] == "structured_data":
        return None, {}

    if case["question_type"] == "unavailable":
        return not cited, {}

    if not cited:
        return False, {"reason": "no_citations"}

    document_rows = documents.set_index("document_id")

    # 1. Every citation belongs to the requested customer.
    belongs = all(
        document_id in document_rows.index
        and document_rows.loc[document_id, "customer_id"]
        == case["customer_id"]
        for document_id in cited
    )

    # 2. At least one expected source is cited.
    expected_cited = bool(
        set(case["expected_sources"]) & set(cited)
    )

    # 3. The cited text actually supports the answer.
    supporting = any(
        document_id in document_rows.index
        and support_ratio(
            result["answer"],
            document_rows.loc[document_id, "content"],
        ) >= MIN_CITATION_SUPPORT
        for document_id in cited
    )

    return belongs and expected_cited and supporting, {
        "belongs_to_customer": belongs,
        "expected_source_cited": expected_cited,
        "citation_supports_answer": supporting,
    }


# ==============================================================
# GROUNDING
# ==============================================================

def evaluate_grounding(
    case: dict,
    result: dict,
) -> tuple[bool, dict]:
    """
    Unavailable: grounded must be False, no sources, exact
    fallback.

    Answerable: grounded must be True AND the answer has to be
    supported by the evidence that was actually in the prompt
    (structured data + retrieved documents): every number in the
    answer appears in the evidence, and most content words do.
    """

    answer = result.get("answer", "") or ""

    if case["question_type"] == "unavailable":
        return (
            result.get("grounded") is False
            and not result.get("sources")
            and answer.strip() == FALLBACK_MESSAGE
        ), {}

    if not result.get("grounded"):
        return False, {"reason": "assistant_marked_not_grounded"}

    context = result.get("context") or {}

    evidence = "\n".join(
        [context.get("structured", "")]
        + [
            document["content"]
            for document in context.get("documents", [])
        ]
    )

    evidence_numbers = [
        value
        for value, _ in extract_numbers(evidence)
    ]

    unsupported_numbers = [
        value
        for value, is_percentage in extract_numbers(answer)
        if not number_supported(
            value,
            is_percentage,
            evidence_numbers,
        )
    ]

    ratio = support_ratio(answer, evidence)

    return (
        not unsupported_numbers
        and ratio >= MIN_EVIDENCE_SUPPORT
    ), {
        "evidence_support_ratio": round(ratio, 3),
        "unsupported_numbers": unsupported_numbers,
    }


# ==============================================================
# RUN EVALUATION
# ==============================================================

def load_evaluation_cases() -> list[dict]:

    if not EVALUATION_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {EVALUATION_PATH}. "
            f"Run python -m src.rag.build_evaluation_dataset"
        )

    with open(EVALUATION_PATH, encoding="utf-8") as file:
        return json.load(file)


def run_evaluation() -> list[dict]:

    print("\n" + "=" * 70)
    print("CUSTOMER RISK INTELLIGENCE")
    print("RAG EVALUATION")
    print("=" * 70)

    cases = load_evaluation_cases()

    if len(cases) < 15:
        raise ValueError(
            "Minimum 15 evaluation cases required."
        )

    documents = pd.read_csv(DOCUMENTS_PATH)

    assistant = CustomerIntelligenceService()

    results = []

    for index, case in enumerate(cases, start=1):

        print("\n" + "-" * 70)
        print(
            f"[{index}/{len(cases)}] {case['eval_id']} "
            f"({case['question_type']})"
        )
        print(f"Customer: {case['customer_id']}")
        print(f"Question: {case['question']}")

        start_time = time.perf_counter()

        try:

            result = assistant.ask(
                customer_id=case["customer_id"],
                question=case["question"],
                include_context=True,
            )

        except Exception as exc:

            result = {
                "answer": "",
                "status": "exception",
                "sources": [],
                "grounded": False,
                "retrieved_document_ids": [],
                "error": f"{type(exc).__name__}: {exc}",
            }

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        retrieval_correct = evaluate_retrieval(case, result)

        answer_correct, answer_details = evaluate_answer(
            case,
            result,
        )

        citation_correct, citation_details = evaluate_citations(
            case,
            result,
            documents,
        )

        grounding_correct, grounding_details = (
            evaluate_grounding(case, result)
        )

        results.append(
            {
                "eval_id": case["eval_id"],
                "customer_id": case["customer_id"],
                "question": case["question"],
                "question_type": case["question_type"],
                "intentionally_unavailable": case[
                    "intentionally_unavailable"
                ],
                "expected_answer": case["expected_answer"],
                "actual_answer": result.get("answer", ""),
                "status": result.get("status"),
                "expected_sources": case["expected_sources"],
                "retrieved_document_ids": result.get(
                    "retrieved_document_ids",
                    [],
                ),
                "actual_sources": result.get("sources", []),
                "assistant_grounded_flag": result.get(
                    "grounded"
                ),
                "retrieval_correct": retrieval_correct,
                "answer_correct": answer_correct,
                "citation_correct": citation_correct,
                "grounding_correct": grounding_correct,
                "details": {
                    "answer": answer_details,
                    "citation": citation_details,
                    "grounding": grounding_details,
                },
                "latency_ms": round(latency_ms, 2),
                "error": result.get("error"),
            }
        )

        def label(value):
            if value is None:
                return "N/A"
            return "PASS" if value else "FAIL"

        print(f"Retrieval: {label(retrieval_correct)}")
        print(f"Answer:    {label(answer_correct)}")
        print(f"Citation:  {label(citation_correct)}")
        print(f"Grounding: {label(grounding_correct)}")
        print(f"Latency:   {latency_ms:.0f} ms")

    return results


# ==============================================================
# SUMMARY
# ==============================================================

METRIC_COLUMNS = {
    "Retrieval accuracy": "retrieval_correct",
    "Answer correctness": "answer_correct",
    "Citation correctness": "citation_correct",
    "Grounding": "grounding_correct",
}


def build_summary(results: list[dict]) -> pd.DataFrame:

    df = pd.DataFrame(results)

    rows = []

    for name, column in METRIC_COLUMNS.items():

        applicable = df[column].dropna().astype(bool)

        rows.append(
            {
                "metric": name,
                "value": round(applicable.mean() * 100, 2),
                "unit": "%",
                "passed": int(applicable.sum()),
                "cases": int(len(applicable)),
            }
        )

    # Hallucination rate: unavailable questions that got
    # anything other than the exact fallback.
    unavailable = df[df["question_type"] == "unavailable"]

    rows.append(
        {
            "metric": "Hallucination rate (unavailable cases)",
            "value": round(
                (~unavailable["answer_correct"]).mean() * 100,
                2,
            ),
            "unit": "%",
            "passed": int((~unavailable["answer_correct"]).sum()),
            "cases": int(len(unavailable)),
        }
    )

    for name, value in [
        ("Average latency", df["latency_ms"].mean()),
        ("Median latency", df["latency_ms"].median()),
        ("P95 latency", df["latency_ms"].quantile(0.95)),
    ]:
        rows.append(
            {
                "metric": name,
                "value": round(value, 1),
                "unit": "ms",
                "passed": None,
                "cases": int(len(df)),
            }
        )

    return pd.DataFrame(rows)


def save_results(
    results: list[dict],
    summary: pd.DataFrame,
) -> None:

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            {
                "summary": summary.to_dict(orient="records"),
                "cases": results,
            },
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    pd.DataFrame(results).drop(
        columns=["details"]
    ).to_csv(
        CASES_CSV_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print(f"\nSaved evaluation results: {OUTPUT_PATH}")
    print(f"Saved per-case CSV:       {CASES_CSV_PATH}")
    print(f"Saved summary CSV:        {SUMMARY_PATH}")


def print_summary(
    results: list[dict],
    summary: pd.DataFrame,
) -> None:

    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("RAG EVALUATION SUMMARY")
    print("=" * 70)

    for _, row in summary.iterrows():

        cases = (
            f"  ({int(row['passed'])}/{int(row['cases'])})"
            if pd.notna(row["passed"])
            else ""
        )

        print(
            f"{row['metric']:<42}"
            f"{row['value']:>9.2f} {row['unit']}{cases}"
        )

    print("\nBy question type (N/A excluded):")

    grouped = (
        df.groupby("question_type")[
            list(METRIC_COLUMNS.values())
        ]
        .agg(
            lambda column: (
                round(column.dropna().astype(bool).mean() * 100, 1)
                if column.notna().any()
                else None
            )
        )
    )

    print(grouped.to_string())

    failed = df[
        df[list(METRIC_COLUMNS.values())]
        .apply(
            lambda row: any(
                value is False
                for value in row
            ),
            axis=1,
        )
    ]

    print("\nCases with at least one failed check:")

    if failed.empty:
        print("None.")
    else:
        for _, row in failed.iterrows():
            print(
                f"- {row['eval_id']} ({row['question_type']}): "
                f"{row['actual_answer'][:160]}"
            )


# ==============================================================
# MAIN
# ==============================================================

def main():

    results = run_evaluation()

    summary = build_summary(results)

    save_results(results, summary)

    print_summary(results, summary)


if __name__ == "__main__":
    main()
