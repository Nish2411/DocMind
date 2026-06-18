import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List
import os


def get_conn():
    return psycopg2.connect(os.getenv("DATABASE_URL"))


def save_document_and_chunks(
    filename: str,
    file_type: str,
    chunks: List[str],
    embeddings: List[List[float]],
) -> str:
    """Insert document + all its chunks into the DB. Returns doc_id."""
    import uuid
    doc_id = str(uuid.uuid4())

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Insert document record
            cur.execute(
                """
                INSERT INTO documents (id, filename, file_type, chunk_count)
                VALUES (%s, %s, %s, %s)
                """,
                (doc_id, filename, file_type, len(chunks))
            )

            # Insert each chunk with its embedding
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vec_str = "[" + ",".join(str(x) for x in embedding) + "]"
                cur.execute(
                    """
                    INSERT INTO chunks (doc_id, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s::vector)
                    """,
                    (doc_id, idx, chunk, vec_str)
                )
        conn.commit()

    return doc_id


def retrieve_similar_chunks(query_embedding: List[float], top_k: int = 6) -> List[dict]:
    """Find top-k most similar chunks across all documents."""
    vec_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    c.content,
                    c.chunk_index,
                    d.filename,
                    d.uploaded_at,
                    1 - (c.embedding <=> %s::vector) AS similarity
                FROM chunks c
                JOIN documents d ON d.id = c.doc_id
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (vec_str, vec_str, top_k)
            )
            return [dict(row) for row in cur.fetchall()]


def list_all_documents() -> List[dict]:
    """Return all documents ordered by upload time."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, filename, file_type, uploaded_at, chunk_count
                FROM documents
                ORDER BY uploaded_at DESC
                """
            )
            rows = [dict(r) for r in cur.fetchall()]

    for row in rows:
        row["uploaded_at"] = row["uploaded_at"].isoformat()

    return rows


def delete_document_by_id(doc_id: str) -> bool:
    """Delete a document and cascade-delete its chunks. Returns True if found."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
            deleted = cur.rowcount > 0
        conn.commit()
    return deleted
    