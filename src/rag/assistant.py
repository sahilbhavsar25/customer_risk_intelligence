import json
import sys

from openai import OpenAI

from src.config.settings import settings
from src.rag.retriever import CustomerDocumentRetriever


# ==============================================================
# CONFIGURATION
# ==============================================================

LLM_MODEL = "gpt-4o-mini"

FALLBACK_MESSAGE = (
    "I could not find sufficient information in the "
    "available customer data."
)


# ==============================================================
# CUSTOMER INTELLIGENCE ASSISTANT
# ==============================================================

class CustomerIntelligenceAssistant:
    """
    Grounded RAG assistant.

    Flow:

        Question
           ↓
        Customer-specific retrieval
           ↓
        Evidence validation
           ↓
        LLM
           ↓
        Grounded response + citations
    """

    def __init__(self):

        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not configured."
            )

        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY
        )

        self.retriever = (
            CustomerDocumentRetriever(
                top_k=5,
                min_similarity=0.35,
            )
        )

    # ==========================================================
    # BUILD CONTEXT
    # ==========================================================

    def build_context(
        self,
        results: list[dict],
    ) -> str:

        context_parts = []

        for index, result in enumerate(
            results,
            start=1,
        ):

            context_parts.append(
                f"""
SOURCE {index}
Customer ID: {result['customer_id']}
Document ID: {result['document_id']}
Document Type: {result['document_type']}
Document Date: {result['document_date']}
Source: {result['source']}
Similarity Score: {result['score']:.4f}

Content:
{result['content']}
""".strip()
            )

        return "\n\n".join(
            context_parts
        )

    # ==========================================================
    # ASK
    # ==========================================================

    def ask(
        self,
        customer_id: str,
        question: str,
    ) -> dict:

        # ------------------------------------------------------
        # Retrieve customer-specific evidence.
        # ------------------------------------------------------

        results = self.retriever.retrieve(
            customer_id=customer_id,
            query=question,
        )

        # ------------------------------------------------------
        # No evidence.
        #
        # IMPORTANT:
        # We do not call the LLM when there is no evidence.
        # ------------------------------------------------------

        if not results:

            return {
                "customer_id": customer_id,
                "question": question,
                "answer": FALLBACK_MESSAGE,
                "sources": [],
                "grounded": False,
            }

        # ------------------------------------------------------
        # Build context from retrieved evidence.
        # ------------------------------------------------------

        context = self.build_context(
            results
        )

        # ------------------------------------------------------
        # System prompt.
        #
        # The model is explicitly forbidden from using
        # information outside the supplied context.
        # ------------------------------------------------------

        system_prompt = f"""
You are a Customer Intelligence Assistant.

Your job is to answer questions about a specific customer
using ONLY the provided customer evidence.

STRICT GROUNDING RULES:

1. Use only the information in the provided SOURCES.
2. Do not invent, infer, or assume facts that are not supported.
3. Do not use general world knowledge to fill missing information.
4. If the sources do not sufficiently answer the question,
   return exactly:
   "{FALLBACK_MESSAGE}"
5. Never mix information from different customers.
6. Every factual statement must be supported by one or more
   source documents.
7. Return citations using the document IDs provided in the
   sources.

Return valid JSON with exactly these fields:

{{
    "answer": "grounded answer",
    "sources": [
        {{
            "document_id": "DOC_000001",
            "document_type": "PAYMENT_ISSUE",
            "document_date": "2025-10-08",
            "source": "ACCOUNT_MANAGEMENT"
        }}
    ],
    "grounded": true
}}

If sufficient evidence is not available, return:

{{
    "answer": "{FALLBACK_MESSAGE}",
    "sources": [],
    "grounded": false
}}

CUSTOMER ID:
{customer_id}

CUSTOMER EVIDENCE:
{context}
""".strip()

        # ------------------------------------------------------
        # User question.
        # ------------------------------------------------------

        user_prompt = (
            f"Customer: {customer_id}\n"
            f"Question: {question}"
        )

        # ------------------------------------------------------
        # Call OpenAI.
        # ------------------------------------------------------

        try:

            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                temperature=0,
                response_format={
                    "type": "json_object"
                },
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )

        except Exception as exc:

            return {
                "customer_id": customer_id,
                "question": question,
                "answer": FALLBACK_MESSAGE,
                "sources": [],
                "grounded": False,
                "error": (
                    "LLM request failed: "
                    f"{str(exc)}"
                ),
            }

        # ------------------------------------------------------
        # Extract response content.
        # ------------------------------------------------------

        content = (
            response.choices[0]
            .message
            .content
        )

        if not content:

            return {
                "customer_id": customer_id,
                "question": question,
                "answer": FALLBACK_MESSAGE,
                "sources": [],
                "grounded": False,
                "error": "Empty LLM response.",
            }

        # ------------------------------------------------------
        # Parse structured JSON.
        # ------------------------------------------------------

        try:

            result = json.loads(
                content
            )

        except json.JSONDecodeError:

            return {
                "customer_id": customer_id,
                "question": question,
                "answer": FALLBACK_MESSAGE,
                "sources": [],
                "grounded": False,
                "error": (
                    "LLM returned invalid JSON."
                ),
            }

        # ------------------------------------------------------
        # Defensive normalization.
        # ------------------------------------------------------

        answer = result.get(
            "answer"
        )

        sources = result.get(
            "sources",
            [],
        )

        grounded = result.get(
            "grounded",
            False,
        )

        if not answer:
            answer = FALLBACK_MESSAGE
            grounded = False
            sources = []

        # ------------------------------------------------------
        # If LLM says fallback, don't allow citations to remain.
        # ------------------------------------------------------

        if answer.strip() == FALLBACK_MESSAGE:

            sources = []
            grounded = False

        return {
            "customer_id": customer_id,
            "question": question,
            "answer": answer,
            "sources": sources,
            "grounded": bool(
                grounded
            ),
        }


# ==============================================================
# CLI TEST
# ==============================================================

def main():

    print("\n" + "=" * 70)
    print("CUSTOMER RISK INTELLIGENCE")
    print("GROUNDED RAG ASSISTANT")
    print("=" * 70)

    customer_id = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "CUST_00842"
    )

    question = (
        " ".join(sys.argv[2:])
        if len(sys.argv) > 2
        else (
            "What payment problems or payment issues "
            "has this customer experienced?"
        )
    )

    print(
        f"\nCustomer: {customer_id}"
    )

    print(
        f"Question: {question}"
    )

    assistant = (
        CustomerIntelligenceAssistant()
    )

    result = assistant.ask(
        customer_id=customer_id,
        question=question,
    )

    print("\n" + "-" * 70)
    print("ANSWER")
    print("-" * 70)

    print(
        result["answer"]
    )

    print("\n" + "-" * 70)
    print("GROUNDED")
    print("-" * 70)

    print(
        result["grounded"]
    )

    print("\n" + "-" * 70)
    print("SOURCES")
    print("-" * 70)

    if result["sources"]:

        for source in result["sources"]:

            print(
                f"Document ID: "
                f"{source.get('document_id')}"
            )

            print(
                f"Document Type: "
                f"{source.get('document_type')}"
            )

            print(
                f"Document Date: "
                f"{source.get('document_date')}"
            )

            print(
                f"Source: "
                f"{source.get('source')}"
            )

            print()

    else:

        print(
            "No sources."
        )

    if result.get("error"):

        print("\n" + "-" * 70)
        print("ERROR")
        print("-" * 70)

        print(
            result["error"]
        )

    print("\n" + "=" * 70)
    print("GROUNDED RAG ASSISTANT TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
