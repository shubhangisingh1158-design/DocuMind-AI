"""
vector_store.py
===============
Step 4 of the "Chat With Your PDF" backend: searching the embeddings.

WHAT THIS MODULE IS FOR
-----------------------
`embedding_manager` turned every chunk into 384 numbers. Those numbers are
only useful if we can ask "which chunks are closest in meaning to this
question?" and get an answer quickly. That is what a vector index does, and
FAISS (Facebook AI Similarity Search) is the library that provides one.

THE ONE INVARIANT THAT MATTERS
------------------------------
FAISS stores only numbers. It does not store text, chunk IDs, or page
numbers - it gives back *row positions* like "row 7 is the best match".

So we keep two things side by side:

    index      -> the vectors, held by FAISS
    metadata   -> a plain Python list of the chunk information

Row N of the index always describes metadata[N]. Everything in this module
exists to keep that promise. If those two ever drifted apart, the app would
confidently quote the wrong page.

WHY INNER PRODUCT (IndexFlatIP)
-------------------------------
`embedding_manager.embed_chunks()` normalises every vector to length 1 by
default. For vectors of length 1, the inner product (dot product) is
exactly the cosine similarity - the standard measure of "how similar in
meaning". So IndexFlatIP gives us cosine similarity scores between -1 and
1, where 1.0 means identical and 0 means unrelated.

"Flat" means exact, brute-force search: every stored vector is compared.
For a PDF's worth of chunks that is instant, and it is always exactly
right. The approximate index types (IVF, HNSW) exist for millions of
vectors and would only add complexity here.

This module does NOT talk to Gemini and draws no interface.

INPUT CONTRACT (produced by src/embedding_manager.py)
-----------------------------------------------------
A list of dictionaries, one per chunk::

    [
        {"chunk_id": "p1-c0", "chunk_index": 0, "page_number": 1,
         "text": "Alice was beginning...",
         "embedding": array([...], dtype=float32), "embedding_dim": 384},
    ]

Run this file directly to try it out:
    python src/vector_store.py sample_pdfs/sample.pdf
"""

from __future__ import annotations

import faiss
import numpy as np

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------

# How many results a search returns when you do not say.
DEFAULT_TOP_K = 5

# If a query vector's length differs from 1 by more than this, we rescale
# it. Normalised vectors are what make the scores true cosine similarity.
NORM_TOLERANCE = 0.01


class VectorStoreError(Exception):
    """
    Raised when an index cannot be built or searched.

    Consistent with PdfProcessingError, ChunkingError and EmbeddingError:
    one error type per module, with a message written for a person.
    """


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _as_float32_matrix(vectors) -> np.ndarray:
    """
    Turn whatever we were given into a 2-D float32 array.

    FAISS is strict: it needs a contiguous 2-D array of float32. A 1-D
    vector of shape (384,) is reshaped to (1, 384) - "one row of 384".
    """
    array = np.asarray(vectors, dtype=np.float32)

    if array.ndim == 1:
        array = array.reshape(1, -1)

    if array.ndim != 2:
        raise VectorStoreError(
            f"Expected a 1-D or 2-D array of numbers, but got "
            f"{array.ndim} dimensions with shape {array.shape}."
        )

    # np.ascontiguousarray guarantees the memory layout FAISS expects.
    return np.ascontiguousarray(array, dtype=np.float32)


def _check_finite(array: np.ndarray, what: str) -> None:
    """
    Reject NaN ("not a number") and infinity.

    These can appear if an embedding went wrong upstream. FAISS will not
    complain - it will silently return nonsense - so we check here.
    """
    if not np.all(np.isfinite(array)):
        raise VectorStoreError(
            f"The {what} contains NaN or infinite values, so it cannot be "
            "used for search. This usually means something went wrong "
            "while creating the embeddings."
        )


def _validate_store(store) -> None:
    """Check that `store` really is a vector store built by this module."""
    if not isinstance(store, dict):
        raise VectorStoreError(
            f"Expected a vector store dictionary, but got a "
            f"{type(store).__name__}.\n"
            "Build one first with build_vector_store()."
        )

    missing = [k for k in ("index", "metadata", "dimension") if k not in store]
    if missing:
        raise VectorStoreError(
            f"This does not look like a vector store - missing key(s): "
            f"{', '.join(missing)}.\n"
            "Build one with build_vector_store()."
        )

    # The invariant described at the top of this file.
    if store["index"].ntotal != len(store["metadata"]):
        raise VectorStoreError(
            "The index and its metadata are out of step "
            f"({store['index'].ntotal} vectors but "
            f"{len(store['metadata'])} metadata entries).\n"
            "Rebuild the store with build_vector_store()."
        )


# ----------------------------------------------------------------------
# Building the index
# ----------------------------------------------------------------------

def create_index(dimension: int):
    """
    Create an empty FAISS index for vectors of the given size.

    Parameters
    ----------
    dimension:
        How many numbers each vector has. Get this from
        `embedding_manager.get_embedding_dimension()` (384 for
        all-MiniLM-L6-v2) rather than hard-coding it.

    Returns
    -------
    An empty faiss.IndexFlatIP. "IP" is inner product, which equals cosine
    similarity for normalised vectors - see the note at the top of this
    file.

    Raises
    ------
    VectorStoreError
        If the dimension is not a positive whole number.

    Notes
    -----
    IndexFlatIP needs no training and no tuning: it is ready to use the
    moment it is created.
    """
    if not isinstance(dimension, (int, np.integer)) or isinstance(dimension, bool):
        raise VectorStoreError(
            f"dimension must be a whole number, got "
            f"{type(dimension).__name__}."
        )

    if dimension <= 0:
        raise VectorStoreError(
            f"dimension must be greater than 0, got {dimension}."
        )

    return faiss.IndexFlatIP(int(dimension))


def build_vector_store(embedded_chunks: list[dict]) -> dict:
    """
    Build a searchable store from embedded chunks.

    The returned store is a plain dictionary holding the FAISS index and a
    parallel list of chunk information. Keeping them in one object is what
    stops them drifting apart.

    Parameters
    ----------
    embedded_chunks:
        The list returned by `embedding_manager.embed_chunks()`. Each item
        needs "chunk_id", "page_number", "text" and "embedding".

    Returns
    -------
    A dictionary::

        {
            "index": <faiss.IndexFlatIP>,   # the vectors
            "metadata": [                   # row N describes metadata[N]
                {"chunk_id": "p1-c0", "page_number": 1,
                 "text": "...", "chunk_index": 0},
                ...
            ],
            "dimension": 384,
            "size": 23,
            "metric": "cosine",
        }

    Raises
    ------
    VectorStoreError
        If the list is empty, an item is the wrong shape, or the vectors
        are not all the same length.
    """
    if not embedded_chunks:
        raise VectorStoreError(
            "No embedded chunks were provided, so there is nothing to "
            "index.\n"
            "Run embedding_manager.embed_chunks() first and pass its "
            "result here."
        )

    required = ("chunk_id", "page_number", "text", "embedding")
    vectors: list[np.ndarray] = []
    metadata: list[dict] = []

    for position, chunk in enumerate(embedded_chunks):
        if not isinstance(chunk, dict):
            raise VectorStoreError(
                f"Item {position} is a {type(chunk).__name__}, but a "
                "dictionary was expected.\n"
                "Pass the list returned by "
                "embedding_manager.embed_chunks()."
            )

        absent = [key for key in required if key not in chunk]
        if absent:
            raise VectorStoreError(
                f"Item {position} is missing the key(s): "
                f"{', '.join(absent)}.\n"
                "Each item must look like "
                '{"chunk_id": "p1-c0", "page_number": 1, "text": "...", '
                '"embedding": array([...])}.'
            )

        vector = np.asarray(chunk["embedding"], dtype=np.float32).reshape(-1)
        vectors.append(vector)

        # Store everything EXCEPT the embedding. The numbers already live
        # in the index; repeating them here would just waste memory.
        entry = {
            "chunk_id": chunk["chunk_id"],
            "page_number": chunk["page_number"],
            "text": chunk["text"],
        }
        if "chunk_index" in chunk:
            entry["chunk_index"] = chunk["chunk_index"]
        metadata.append(entry)

    # Every vector must be the same length or the index is meaningless.
    lengths = {vector.shape[0] for vector in vectors}
    if len(lengths) != 1:
        raise VectorStoreError(
            f"The embeddings are not all the same length: found sizes "
            f"{sorted(lengths)}.\n"
            "All chunks must be embedded by the same model."
        )

    dimension = lengths.pop()
    matrix = _as_float32_matrix(np.vstack(vectors))
    _check_finite(matrix, "chunk embeddings")

    index = create_index(dimension)
    index.add(matrix)

    # Belt and braces: confirm the invariant right after building.
    if index.ntotal != len(metadata):
        raise VectorStoreError(
            "Internal error: the index and metadata ended up different "
            f"sizes ({index.ntotal} vs {len(metadata)})."
        )

    return {
        "index": index,
        "metadata": metadata,
        "dimension": int(dimension),
        "size": int(index.ntotal),
        "metric": "cosine",
    }


# ----------------------------------------------------------------------
# Searching
# ----------------------------------------------------------------------

def search(
    store: dict,
    query_embedding,
    top_k: int = DEFAULT_TOP_K,
    normalize_query: bool = True,
) -> list[dict]:
    """
    Find the chunks whose meaning is closest to a query embedding.

    Parameters
    ----------
    store:
        A store built by `build_vector_store()`.
    query_embedding:
        One vector, as returned by `embedding_manager.embed_query()`.
        Shape (384,) or (1, 384) - both are accepted.
    top_k:
        How many results to return. If it is larger than the number of
        chunks in the store, every chunk is returned (no error).
    normalize_query:
        Rescale the query vector to length 1 if it is not already. This
        keeps the scores true cosine similarity. Leave it True unless you
        know you want raw inner products.

    Returns
    -------
    A list of dictionaries, best match first::

        {
            "rank": 1,              # 1 = best
            "score": 0.6241,        # cosine similarity, -1.0 to 1.0
            "chunk_id": "p3-c0",
            "page_number": 3,
            "text": "passage into the loveliest garden...",
            "row": 5,               # which row of the index matched
            "chunk_index": 5,       # present if the chunker provided it
        }

    Raises
    ------
    VectorStoreError
        If the store is empty or malformed, the query is the wrong size,
        or top_k is not a positive whole number.
    """
    _validate_store(store)

    index = store["index"]
    metadata = store["metadata"]

    if index.ntotal == 0:
        raise VectorStoreError(
            "This store is empty, so there is nothing to search.\n"
            "Build it from embedded chunks with build_vector_store()."
        )

    if not isinstance(top_k, (int, np.integer)) or isinstance(top_k, bool):
        raise VectorStoreError(
            f"top_k must be a whole number, got {type(top_k).__name__}."
        )

    if top_k <= 0:
        raise VectorStoreError(
            f"top_k must be greater than 0, got {top_k}."
        )

    query = _as_float32_matrix(query_embedding)

    if query.shape[0] != 1:
        raise VectorStoreError(
            f"Expected a single query vector, but got {query.shape[0]} "
            "rows.\nSearch one question at a time."
        )

    if query.shape[1] != store["dimension"]:
        raise VectorStoreError(
            f"The query has {query.shape[1]} numbers but this store holds "
            f"vectors of {store['dimension']} numbers.\n"
            "The question must be embedded with the same model as the "
            "chunks - see embedding_manager.embed_query()."
        )

    _check_finite(query, "query embedding")

    if normalize_query:
        length = float(np.linalg.norm(query))
        if length == 0.0:
            raise VectorStoreError(
                "The query embedding is all zeros, so it has no direction "
                "to compare against."
            )
        if abs(length - 1.0) > NORM_TOLERANCE:
            query = query / length

    # Asking for more results than we have would make FAISS pad the output
    # with -1 row numbers, so clamp instead.
    wanted = min(int(top_k), index.ntotal)

    # FAISS returns two arrays: scores and row positions, best first.
    scores, rows = index.search(query, wanted)

    results: list[dict] = []
    for rank, (score, row) in enumerate(zip(scores[0], rows[0]), start=1):
        # -1 means "no result in this slot". It should not happen now that
        # we clamp top_k, but checking costs nothing.
        if row < 0:
            continue

        entry = metadata[int(row)]
        result = {
            "rank": rank,
            "score": round(float(score), 4),
            "chunk_id": entry["chunk_id"],
            "page_number": entry["page_number"],
            "text": entry["text"],
            "row": int(row),
        }
        if "chunk_index" in entry:
            result["chunk_index"] = entry["chunk_index"]
        results.append(result)

    return results


def search_by_text(
    store: dict,
    question: str,
    top_k: int = DEFAULT_TOP_K,
    **embed_kwargs,
) -> list[dict]:
    """
    Convenience wrapper: embed a question, then search with it.

    This saves you from remembering to embed the question with the same
    model as the chunks - the single easiest mistake to make here, and one
    that produces quietly wrong results rather than an error.

    Parameters
    ----------
    store:
        A store built by `build_vector_store()`.
    question:
        The question, as plain text.
    top_k:
        How many results to return.
    **embed_kwargs:
        Passed straight through to `embedding_manager.embed_query()`, for
        example `device="mps"`.

    Returns
    -------
    The same list of result dictionaries as `search()`.

    Notes
    -----
    embedding_manager is imported inside this function so that importing
    vector_store does not pull in PyTorch unless you actually need it.
    """
    from embedding_manager import embed_query

    query_embedding = embed_query(question, **embed_kwargs)
    return search(store, query_embedding, top_k=top_k)


# ----------------------------------------------------------------------
# Inspecting a store
# ----------------------------------------------------------------------

def summarise_store(store: dict) -> dict:
    """
    Describe a store without printing any vectors.

    Returns
    -------
    A dictionary like::

        {
            "size": 23,
            "dimension": 384,
            "metric": "cosine",
            "pages_covered": 11,
            "index_type": "IndexFlatIP",
            "is_consistent": True,
        }
    """
    _validate_store(store)

    return {
        "size": int(store["index"].ntotal),
        "dimension": int(store["dimension"]),
        "metric": store.get("metric", "cosine"),
        "pages_covered": len({e["page_number"] for e in store["metadata"]}),
        "index_type": type(store["index"]).__name__,
        "is_consistent": store["index"].ntotal == len(store["metadata"]),
    }


# ----------------------------------------------------------------------
# Runnable test
#
# Uses the real project modules end to end:
#   pdf_processor -> text_chunker -> embedding_manager -> vector_store
# Only runs when you execute this file directly.
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from embedding_manager import EmbeddingError, embed_chunks, embed_query
    from pdf_processor import PdfProcessingError, extract_text_from_pdf
    from text_chunker import chunk_pages

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample_pdfs/sample.pdf"

    print(f"\nPDF: {pdf_path}")

    # --- 1. read ------------------------------------------------------
    try:
        pages = extract_text_from_pdf(pdf_path)
    except PdfProcessingError as error:
        print(f"\nCould not process that PDF:\n{error}\n")
        sys.exit(1)
    print(f"  pages read     : {len(pages)}")

    # --- 2. chunk -----------------------------------------------------
    chunks = chunk_pages(pages)
    print(f"  chunks created : {len(chunks)}")
    if not chunks:
        print("\nNo text found in that PDF, so there is nothing to index.\n")
        sys.exit(0)

    # --- 3. embed -----------------------------------------------------
    print("  embedding      : (first run downloads about 90 MB)")
    try:
        embedded = embed_chunks(chunks)
    except EmbeddingError as error:
        print(f"\nEmbedding failed:\n{error}\n")
        sys.exit(1)

    # --- 4. index -----------------------------------------------------
    started = time.perf_counter()
    store = build_vector_store(embedded)
    build_seconds = time.perf_counter() - started

    stats = summarise_store(store)
    print(f"\nIndex built in {build_seconds * 1000:.1f} ms")
    print(f"  index type     : {stats['index_type']}")
    print(f"  vectors stored : {stats['size']}")
    print(f"  dimension      : {stats['dimension']}")
    print(f"  metric         : {stats['metric']}")
    print(f"  pages covered  : {stats['pages_covered']}")
    print(f"  consistent     : {stats['is_consistent']}")

    # --- 5. search ----------------------------------------------------
    questions = [
        "What did Alice find on the little glass table?",
        "a mouse swimming in a pool of tears",
        "the white rabbit with a pocket watch",
    ]

    for question in questions:
        print(f"\nQ: {question}")
        try:
            matches = search_by_text(store, question, top_k=3)
        except (VectorStoreError, EmbeddingError) as error:
            print(f"   search failed: {error}")
            continue

        for match in matches:
            preview = match["text"][:68].replace("\n", " ")
            print(f"   {match['rank']}. score {match['score']:+.4f}  "
                  f"{match['chunk_id']:<7} page {match['page_number']:<3} "
                  f"\"{preview}...\"")

    print("\nDone.\n")
