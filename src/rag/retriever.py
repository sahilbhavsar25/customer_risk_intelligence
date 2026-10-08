import logging
import re
import sys
from dataclasses import dataclass, field

from langchain_core.documents import Document

from qdrant_client import QdrantClient, models

from src.config.settings import settings


logger = logging.getLogger(__name__)


# ==============================================================
# CONFIGURATION
# ==============================================================

COLLECTION_NAME = "customer_risk_documents"

EMBEDDING_MODEL = "text-embedding-3-small"

TOP_K = 4

MIN_SIMILARITY_SCORE = 0.35


# ==============================================================
# QUESTION -> DOCUMENT TYPE MAPPING
# ==============================================================

DOCUMENT_TYPE_KEYWORDS = {
    "PAYMENT_ISSUE": [
        "payment",
        "payments",
        "invoice",
        "invoices",
        "outstanding amount",
        "overdue payment",
        "payment delay",
        "payment problem",
        "payment problems",
        "payment issue",
        "payment issues",
        "due amount",
        "billing",
    ],
    "COMPLAINT": [
        "complaint",
        "complaints",
        "complain",
        "complained",
        "dissatisfaction",
        "dissatisfied",
        "unhappy",
    ],
    "SUPPORT_NOTE": [
        "support",
        "support issue",
        "technical issue",
        "technical issues",
        "technical problem",
        "technical problems",
        "assistance",
        "troubleshooting",
    ],
    "RENEWAL_DISCUSSION": [
        "renewal",
        "renewals",
        "renew",
        "subscription renewal",
        "contract renewal",
    ],
    "ESCALATION": [
        "escalation",
        "escalations",
        "escalated",
    ],
    "ACCOUNT_NOTE": [
        "account note",
        "account notes",
        "account review",
        "account issue",
        "account issues",
        "account concern",
        "account concerns",
        "account-related",
        "account management",
    ],
    "CUSTOMER_PREFERENCE": [
        "preference",
        "preferences",
        "prefer",
        "prefers",
        "preferred",
        "communication channel",
        "contact method",
    ],
}


# ==============================================================
# QUERY EXPANSION
#
# Short questions ("What customer preference was recorded?")
# embed poorly against note-style documents. When a document
# type is detected confidently, the embedded query is extended
# with vocabulary describing that type. This only changes how
# the query is embedded - it adds no facts, and the answer is
# still produced from the retrieved text only.
# ==============================================================

QUERY_EXPANSIONS = {
    "PAYMENT_ISSUE": (
        "payment delayed, outstanding amount, overdue invoice, "
        "payment status, payment timeline, finance team"
    ),
    "COMPLAINT": (
        "customer complaint, dissatisfaction, service issues, "
        "delayed service response, requested resolution"
    ),
    "SUPPORT_NOTE": (
        "support interaction, requested assistance, technical "
        "issue, troubleshooting guidance, support team"
    ),
    "RENEWAL_DISCUSSION": (
        "renewal discussion, renewal terms, pricing, contract "
        "terms, service continuity"
    ),
    "ESCALATION": (
        "case escalated, senior support team, additional "
        "investigation, issue not resolved"
    ),
    "ACCOUNT_NOTE": (
        "account review, account management team, customer "
        "profile, recent account activity"
    ),
    "CUSTOMER_PREFERENCE": (
        "communication preference, prefers email, online "
        "portal, account updates"
    ),
}

# Broad risk questions ("What evidence supports the current
# risk?") share almost no words with the notes themselves
# (raw similarity ~0.30). Describing what risk evidence looks
# like brings relevant notes above the threshold (~0.42).
GENERAL_QUERY_EXPANSION = (
    "risk signals: payment problems, overdue invoices, "
    "complaints, dissatisfaction, escalations, unresolved "
    "support issues, renewal concerns"
)


# ==============================================================
# TOPICS WE HAVE NO DATA FOR
#
# Questions about these must not trigger a broad semantic
# search - the nearest neighbour of "legal dispute" is still
# *some* document, and handing it to the LLM creates false
# evidence.
# ==============================================================

UNSUPPORTED_TOPIC_KEYWORDS = [
    "legal",
    "lawsuit",
    "litigation",
    "court",
    "attorney",
    "lawyer",
    "credit score",
    "credit rating",
    "bankruptcy",
    "insolvency",
    "revenue",
    "profit",
    "headcount",
    "employees",
    "competitor",
    "competitors",
    "merger",
    "acquisition",
    "fraud",
    "data breach",
    "security breach",
    "password",
    "home address",
    "phone number",
    "social media",
]


# ==============================================================
# GENERAL QUESTIONS
#
# Questions without a specific document type that still
# legitimately need all of the customer's documents.
# ==============================================================

GENERAL_QUERY_KEYWORDS = [
    "risk",
    "risky",
    "evidence",
    "history",
    "summary",
    "summarize",
    "summarise",
    "overview",
    "concern",
    "concerns",
    "red flag",
    "red flags",
    "documents",
    "notes",
    "recent activity",
    "what happened",
    "churn",
]


# ==============================================================
# QUERY PLAN
# ==============================================================

@dataclass
class QueryPlan:
    """
    mode:
        typed        -> search only the detected document type(s)
        general      -> search all of the customer's documents
        unsupported  -> topic we hold no data for, no search
        none         -> no document topic detected, no search
    """

    mode: str
    document_types: list[str] = field(
        default_factory=list
    )

    @property
    def searches_documents(self) -> bool:
        return self.mode in {"typed", "general"}


def keyword_matches(
    query: str,
    keyword: str,
) -> bool:
    """
    Word-boundary match.

    'support' matches "support" but not "supports" /
    "supported", which prevents "What evidence supports the
    risk?" from being routed to SUPPORT_NOTE.
    """

    pattern = (
        rf"(?<![a-z0-9]){re.escape(keyword.lower())}"
        rf"(?![a-z0-9])"
    )

    return bool(
        re.search(
            pattern,
            query.lower(),
        )
    )


def plan_query(query: str) -> QueryPlan:

    if any(
        keyword_matches(query, keyword)
        for keyword in UNSUPPORTED_TOPIC_KEYWORDS
    ):
        return QueryPlan(mode="unsupported")

    matched_types = [
        document_type
        for document_type, keywords in DOCUMENT_TYPE_KEYWORDS.items()
        if any(
            keyword_matches(query, keyword)
            for keyword in keywords
        )
    ]

    if matched_types:
        return QueryPlan(
            mode="typed",
            document_types=matched_types,
        )

    if any(
        keyword_matches(query, keyword)
        for keyword in GENERAL_QUERY_KEYWORDS
    ):
        return QueryPlan(mode="general")

    return QueryPlan(mode="none")


def build_search_text(
    query: str,
    plan: QueryPlan,
) -> str:

    if plan.mode == "general":
        return f"{query} {GENERAL_QUERY_EXPANSION}"

    if plan.mode != "typed":
        return query

    expansions = [
        QUERY_EXPANSIONS[document_type]
        for document_type in plan.document_types
        if document_type in QUERY_EXPANSIONS
    ]

    return " ".join([query] + expansions)


# ==============================================================
# RETRIEVER
# ==============================================================

class CustomerDocumentRetriever:
    """
    Customer-specific semantic retriever.

    Retrieval flow:

        User question
              ↓
        Query plan (typed / general / unsupported / none)
              ↓
        Customer filter (always)
              ↓
        Document-type filter (when detected)
              ↓
        Semantic similarity search (top-k)
              ↓
        Similarity threshold
              ↓
        Relevant evidence, or [] if nothing is good enough
    """

    def __init__(
        self,
        top_k: int = TOP_K,
        min_similarity: float = MIN_SIMILARITY_SCORE,
        client: QdrantClient | None = None,
        embeddings=None,
    ):

        self.top_k = top_k
        self.min_similarity = min_similarity

        # Clients can be injected (tests use fakes).
        if embeddings is None:

            if not settings.OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY is not configured."
                )

            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                api_key=settings.OPENAI_API_KEY,
            )

        self.embeddings = embeddings

        self.client = client or QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
            timeout=10,
        )

        if not self.client.collection_exists(
            COLLECTION_NAME
        ):
            raise ValueError(
                f"Qdrant collection does not exist: "
                f"{COLLECTION_NAME}. Run "
                f"python -m src.rag.ingest_documents"
            )

    # ==========================================================
    # BUILD QDRANT FILTER
    # ==========================================================

    def build_filter(
        self,
        customer_id: str,
        document_types: list[str] | None = None,
    ) -> models.Filter:

        conditions = [
            models.FieldCondition(
                key="metadata.customer_id",
                match=models.MatchValue(
                    value=customer_id
                ),
            )
        ]

        if document_types:

            conditions.append(
                models.FieldCondition(
                    key="metadata.document_type",
                    match=models.MatchAny(
                        any=document_types
                    ),
                )
            )

        return models.Filter(
            must=conditions
        )

    # ==========================================================
    # RETRIEVE
    # ==========================================================

    def retrieve(
        self,
        customer_id: str,
        query: str,
        plan: QueryPlan | None = None,
    ) -> list[dict]:

        customer_id = customer_id.strip()
        query = query.strip()

        if not customer_id:
            raise ValueError(
                "customer_id cannot be empty."
            )

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        plan = plan or plan_query(query)

        logger.info(
            "retrieval plan customer=%s mode=%s types=%s",
            customer_id,
            plan.mode,
            plan.document_types or "ALL",
        )

        # ------------------------------------------------------
        # No search for unsupported / undetected topics.
        # Also saves an embedding call.
        # ------------------------------------------------------

        if not plan.searches_documents:
            return []

        query_vector = self.embeddings.embed_query(
            build_search_text(
                query,
                plan,
            )
        )

        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=self.build_filter(
                customer_id=customer_id,
                document_types=plan.document_types,
            ),
            limit=self.top_k,
            with_payload=True,
        )

        results = {}

        for point in response.points:

            score = float(point.score)

            # Ignore weak matches.
            if score < self.min_similarity:
                continue

            payload = point.payload or {}
            metadata = payload.get("metadata", {})

            # --------------------------------------------------
            # Defensive customer isolation check. The Qdrant
            # filter already guarantees this; a second check
            # costs nothing.
            # --------------------------------------------------

            if metadata.get("customer_id") != customer_id:
                logger.error(
                    "Cross-customer document blocked: "
                    "requested=%s got=%s",
                    customer_id,
                    metadata.get("customer_id"),
                )
                continue

            document_id = metadata.get("document_id")

            # Keep the best-scoring chunk per document.
            if (
                document_id in results
                and results[document_id]["score"] >= score
            ):
                continue

            results[document_id] = {
                "score": score,
                "content": payload.get(
                    "page_content",
                    "",
                ),
                "customer_id": metadata.get(
                    "customer_id"
                ),
                "document_id": document_id,
                "document_type": metadata.get(
                    "document_type"
                ),
                "document_date": metadata.get(
                    "document_date"
                ),
                "source": metadata.get(
                    "source"
                ),
            }

        return sorted(
            results.values(),
            key=lambda result: result["score"],
            reverse=True,
        )

    # ==========================================================
    # LANGCHAIN DOCUMENT FORMAT
    # ==========================================================

    def retrieve_documents(
        self,
        customer_id: str,
        query: str,
    ) -> list[Document]:

        return [
            Document(
                page_content=result["content"],
                metadata={
                    key: value
                    for key, value in result.items()
                    if key != "content"
                },
            )
            for result in self.retrieve(
                customer_id,
                query,
            )
        ]


# ==============================================================
# CLI
# ==============================================================

def main():

    print("\n" + "=" * 70)
    print("CUSTOMER RISK INTELLIGENCE")
    print("RAG RETRIEVER TEST")
    print("=" * 70)

    customer_id = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "CUST_00842"
    )

    query = (
        " ".join(sys.argv[2:])
        if len(sys.argv) > 2
        else (
            "What payment problems or payment issues "
            "has this customer experienced?"
        )
    )

    plan = plan_query(query)

    print(f"\nCustomer: {customer_id}")
    print(f"Query: {query}")
    print(
        f"Plan: {plan.mode} "
        f"{plan.document_types or ''}"
    )

    retriever = CustomerDocumentRetriever()

    results = retriever.retrieve(
        customer_id=customer_id,
        query=query,
        plan=plan,
    )

    print(f"\nRetrieved documents: {len(results)}")

    if not results:
        print("\nNO SUFFICIENT DOCUMENT EVIDENCE FOUND")

    for index, result in enumerate(results, start=1):

        print("\n" + "-" * 70)
        print(f"Result #{index}")
        print(f"Similarity score: {result['score']:.4f}")
        print(f"Document ID: {result['document_id']}")
        print(f"Document type: {result['document_type']}")
        print(f"Document date: {result['document_date']}")
        print(f"Source: {result['source']}")
        print(f"Content:\n{result['content']}")

    print("\n" + "=" * 70)
    print("RAG RETRIEVER TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
