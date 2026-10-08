from pathlib import Path
from typing import List
import hashlib
import sys
import uuid

import pandas as pd

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient, models

from src.config.settings import settings


# ==============================================================
# PATH CONFIGURATION
# ==============================================================

DOCUMENTS_PATH = Path(
    "data/processed/documents/documents_clean.csv"
)

COLLECTION_NAME = "customer_risk_documents"

QDRANT_URL = settings.QDRANT_URL


# ==============================================================
# EMBEDDING CONFIGURATION
# ==============================================================

EMBEDDING_MODEL = "text-embedding-3-small"

# text-embedding-3-small produces 1536 dimensions by default.
VECTOR_SIZE = 1536


# ==============================================================
# CHUNKING CONFIGURATION
# ==============================================================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


# ==============================================================
# BATCH CONFIGURATION
# ==============================================================

BATCH_SIZE = 50


# ==============================================================
# LOAD DOCUMENTS
# ==============================================================

def load_documents() -> pd.DataFrame:

    if not DOCUMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Clean documents file not found: "
            f"{DOCUMENTS_PATH}"
        )

    df = pd.read_csv(
        DOCUMENTS_PATH
    )

    required_columns = {
        "document_id",
        "customer_id",
        "document_type",
        "document_date",
        "source",
        "content",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing document columns: "
            f"{sorted(missing_columns)}"
        )

    # ----------------------------------------------------------
    # Remove unusable rows defensively.
    # ----------------------------------------------------------

    df = df.dropna(
        subset=[
            "document_id",
            "customer_id",
            "content",
        ]
    ).copy()

    df["content"] = (
        df["content"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["content"] != ""
    ].copy()

    print(
        f"Loaded clean documents: "
        f"{len(df):,}"
    )

    return df


# ==============================================================
# CONVERT CSV ROWS TO LANGCHAIN DOCUMENTS
# ==============================================================

def create_langchain_documents(
    df: pd.DataFrame,
) -> List[Document]:

    documents = []

    for _, row in df.iterrows():

        metadata = {
            "customer_id": str(
                row["customer_id"]
            ),
            "document_id": str(
                row["document_id"]
            ),
            "document_type": str(
                row["document_type"]
            ),
            "document_date": str(
                row["document_date"]
            ),
            "source": str(
                row["source"]
            ),
        }

        # Hash of everything that ends up in Qdrant. If it hasn't
        # changed, the document is not re-embedded.
        metadata["content_hash"] = hashlib.sha256(
            (
                str(row["content"])
                + "|"
                + "|".join(
                    metadata[key]
                    for key in sorted(metadata)
                )
            ).encode("utf-8")
        ).hexdigest()

        document = Document(
            page_content=str(
                row["content"]
            ),
            metadata=metadata,
        )

        documents.append(
            document
        )

    print(
        f"Created LangChain documents: "
        f"{len(documents):,}"
    )

    return documents


# ==============================================================
# CHUNK DOCUMENTS
# ==============================================================

def split_documents(
    documents: List[Document],
) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = []

    for document in documents:

        for index, chunk in enumerate(
            splitter.split_documents([document])
        ):
            chunk.metadata["chunk_index"] = index
            chunks.append(chunk)

    print(
        f"Created document chunks: "
        f"{len(chunks):,}"
    )

    return chunks


# ==============================================================
# CREATE QDRANT CLIENT
# ==============================================================

def create_qdrant_client() -> QdrantClient:

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=settings.QDRANT_API_KEY or None,
    )

    return client


# ==============================================================
# CREATE COLLECTION
# ==============================================================

def create_collection(
    client: QdrantClient,
    rebuild: bool = False,
) -> None:

    exists = client.collection_exists(
        COLLECTION_NAME
    )

    # ----------------------------------------------------------
    # Default is incremental: keep the collection and only
    # (re)embed what changed. --rebuild drops everything, e.g.
    # after changing the embedding model or chunking settings.
    # ----------------------------------------------------------

    if exists and rebuild:

        print(
            f"Rebuild requested - deleting collection: "
            f"{COLLECTION_NAME}"
        )

        client.delete_collection(
            collection_name=COLLECTION_NAME
        )

        exists = False

    if not exists:

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

        print(
            f"Created Qdrant collection: "
            f"{COLLECTION_NAME}"
        )

    # Payload indexes for the fields every query filters on.
    for field_name in [
        "metadata.customer_id",
        "metadata.document_type",
        "metadata.document_id",
    ]:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


# ==============================================================
# DIFF AGAINST WHAT IS ALREADY STORED
# ==============================================================

def get_stored_hashes(
    client: QdrantClient,
) -> dict[str, str]:
    """
    document_id -> content_hash for everything in Qdrant.
    """

    stored = {}
    offset = None

    while True:

        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:

            metadata = (
                (point.payload or {})
                .get("metadata", {})
            )

            document_id = metadata.get("document_id")

            if document_id:
                stored[document_id] = metadata.get(
                    "content_hash"
                )

        if offset is None:
            break

    return stored


def delete_documents(
    client: QdrantClient,
    document_ids: list[str],
) -> None:

    if not document_ids:
        return

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.document_id",
                        match=models.MatchAny(
                            any=document_ids
                        ),
                    )
                ]
            )
        ),
    )


def point_id(chunk: Document) -> str:
    """
    Deterministic point ID, so re-upserting the same chunk
    overwrites it instead of creating a duplicate.
    """

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{chunk.metadata['document_id']}:"
            f"{chunk.metadata['chunk_index']}",
        )
    )


# ==============================================================
# EMBED AND STORE CHUNKS
# ==============================================================

def embed_and_store(
    client: QdrantClient,
    chunks: List[Document],
) -> None:

    if not chunks:
        print(
            "No new or changed chunks to embed."
        )
        return

    if not settings.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not configured."
        )

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )

    total_chunks = len(chunks)

    print(
        "\nEmbedding and storing chunks..."
    )

    for start in range(
        0,
        total_chunks,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            total_chunks,
        )

        batch = chunks[
            start:end
        ]

        texts = [
            document.page_content
            for document in batch
        ]

        vectors = embeddings.embed_documents(
            texts
        )

        points = []

        for document, vector in zip(
            batch,
            vectors,
        ):

            payload = {
                "page_content": (
                    document.page_content
                ),
                "metadata": (
                    document.metadata
                ),
            }

            point = models.PointStruct(
                id=point_id(document),
                vector=vector,
                payload=payload,
            )

            points.append(
                point
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        print(
            f"  Stored "
            f"{end:,}/{total_chunks:,} chunks"
        )

    print(
        "\nAll chunks successfully stored."
    )


# ==============================================================
# VERIFY COLLECTION
# ==============================================================

def verify_collection(
    client: QdrantClient,
) -> None:

    collection_info = (
        client.get_collection(
            collection_name=COLLECTION_NAME
        )
    )

    print("\n" + "=" * 70)
    print("QDRANT COLLECTION")
    print("=" * 70)

    print(
        f"Collection: "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Points stored: "
        f"{collection_info.points_count}"
    )

    print(
        f"Vector size: "
        f"{collection_info.config.params.vectors.size}"
    )

    print(
        f"Distance: "
        f"{collection_info.config.params.vectors.distance}"
    )


# ==============================================================
# MAIN
# ==============================================================

def main() -> None:

    rebuild = "--rebuild" in sys.argv

    print("\n" + "=" * 70)
    print("CUSTOMER RISK INTELLIGENCE")
    print("RAG DOCUMENT INGESTION")
    print("=" * 70)

    # ----------------------------------------------------------
    # 1. Load cleaned documents
    # ----------------------------------------------------------

    df = load_documents()

    # ----------------------------------------------------------
    # 2. Convert to LangChain Documents
    # ----------------------------------------------------------

    documents = (
        create_langchain_documents(
            df
        )
    )

    # ----------------------------------------------------------
    # 3. Chunk documents
    # ----------------------------------------------------------

    chunks = split_documents(
        documents
    )

    # ----------------------------------------------------------
    # 4. Connect to Qdrant
    # ----------------------------------------------------------

    print(
        f"\nConnecting to Qdrant: "
        f"{QDRANT_URL}"
    )

    client = create_qdrant_client()

    # Test connection.
    client.get_collections()

    print(
        "[PASS] Qdrant connection"
    )

    # ----------------------------------------------------------
    # 5. Create collection (kept unless --rebuild)
    # ----------------------------------------------------------

    create_collection(
        client,
        rebuild=rebuild,
    )

    # ----------------------------------------------------------
    # 6. Embed only new / changed documents, remove deleted
    # ----------------------------------------------------------

    stored_hashes = get_stored_hashes(client)

    current_hashes = {
        chunk.metadata["document_id"]: chunk.metadata[
            "content_hash"
        ]
        for chunk in chunks
    }

    changed_ids = [
        document_id
        for document_id, content_hash in current_hashes.items()
        if stored_hashes.get(document_id) != content_hash
    ]

    removed_ids = [
        document_id
        for document_id in stored_hashes
        if document_id not in current_hashes
    ]

    print(
        f"\nDocuments unchanged:     "
        f"{len(current_hashes) - len(changed_ids):,}"
    )
    print(
        f"Documents new/changed:   {len(changed_ids):,}"
    )
    print(
        f"Documents removed:       {len(removed_ids):,}"
    )

    # Old chunks of changed docs are removed first, in case the
    # new version has fewer chunks.
    delete_documents(
        client,
        changed_ids + removed_ids,
    )

    changed = set(changed_ids)

    embed_and_store(
        client,
        [
            chunk
            for chunk in chunks
            if chunk.metadata["document_id"] in changed
        ],
    )

    # ----------------------------------------------------------
    # 7. Verify
    # ----------------------------------------------------------

    verify_collection(
        client
    )

    print("\n" + "=" * 70)
    print("RAG DOCUMENT INGESTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
