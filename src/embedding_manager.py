"""
embedding_manager.py
====================
Step 3 of the "Chat With Your PDF" backend: turning chunks into vectors.

WHAT AN EMBEDDING IS
--------------------
An embedding is a list of numbers that represents the *meaning* of a piece
of text. The model that produces it has been trained so that texts with
similar meanings get similar lists of numbers.

That is the trick that makes this project work. Plain keyword search fails
when the wording differs: searching for "car" will not find a page that
only says "automobile". But the embeddings of those two words sit close
together, so comparing numbers finds the match that comparing letters
misses.

The model used here, `all-MiniLM-L6-v2`, turns any piece of text into
384 numbers. Every chunk becomes a point in 384-dimensional space, and
"finding relevant text" becomes "finding nearby points".

This module does NOT build a search index (that is FAISS, a later step),
does not talk to Gemini, and draws no interface. Chunks in, vectors out.

INPUT CONTRACT (produced by src/text_chunker.py)
------------------------------------------------
A list of dictionaries, one per chunk::

    [
        {"chunk_id": "p1-c0", "chunk_index": 0, "page_number": 1,
         "text": "Alice was beginning...", "char_count": 1987,
         "approx_tokens": 500},
    ]

FIRST RUN WARNING
-----------------
The very first time the model is loaded it is downloaded from the internet
(about 90 MB) and cached in ~/.cache/huggingface/. That takes a moment and
needs a connection. Every run after that loads from the cache and works
offline.

Run this file directly to try it out:
    python src/embedding_manager.py sample_pdfs/sample.pdf
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------

# The embedding model. all-MiniLM-L6-v2 is the standard starting choice:
# small (~90 MB), fast, and good quality. It outputs 384 numbers per text.
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# How many chunks to send through the model at once. Larger batches are
# faster but use more memory. 32 is the library's own default and is a
# comfortable size for a laptop.
DEFAULT_BATCH_SIZE = 32

# Normalising scales every vector to length 1. Do not change this unless
# you know why: with normalised vectors, "angle between vectors" (cosine
# similarity) and "dot product" become the same calculation, which is what
# the search step will rely on later.
DEFAULT_NORMALIZE = True


class EmbeddingError(Exception):
    """
    Raised when the model cannot be loaded or the input data is wrong.

    As in the other modules, one custom error type means calling code can
    catch exactly one thing and show the message to the user.
    """


# A loaded model is expensive to create (it reads ~90 MB from disk and
# builds a neural network in memory), so we keep loaded models here and
# hand back the same object on later calls. The key is (model_name, device).
_MODEL_CACHE: dict[tuple[str, str | None], object] = {}


# ----------------------------------------------------------------------
# Loading the model
# ----------------------------------------------------------------------

def load_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
):
    """
    Load the embedding model, reusing it if it is already loaded.

    Call this as often as you like: the first call does the slow work and
    every later call with the same arguments returns the same object
    instantly. This is why the function exists rather than creating a
    SentenceTransformer directly wherever it is needed.

    Parameters
    ----------
    model_name:
        Name of a Sentence Transformers model. Defaults to
        "all-MiniLM-L6-v2".
    device:
        Where to run the model. Leave as None to let the library choose,
        which is the right answer almost always. On an Apple Silicon Mac
        you can pass "mps" to use the GPU, or "cpu" to force the CPU.
        There is no "cuda" on a Mac - that is NVIDIA only.

    Returns
    -------
    A ready-to-use SentenceTransformer object.

    Raises
    ------
    EmbeddingError
        If the model cannot be downloaded or loaded.

    Notes
    -----
    The import of sentence_transformers happens inside this function on
    purpose. That library pulls in PyTorch, which takes a second or two to
    start up, so we pay that cost only when a model is actually needed -
    not every time this file is imported.
    """
    cache_key = (model_name, device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingError(
            "The 'sentence-transformers' package is not available.\n"
            "Activate your virtual environment and install it with:\n"
            "  python -m pip install sentence-transformers"
        ) from exc

    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception as exc:
        raise EmbeddingError(
            f"Could not load the embedding model '{model_name}'.\n"
            "If this is the first run, the model needs to be downloaded "
            "(about 90 MB), so check your internet connection.\n"
            "If you are offline, the model must already be in "
            "~/.cache/huggingface/.\n"
            f"Technical detail: {type(exc).__name__}: {exc}"
        ) from None

    _MODEL_CACHE[cache_key] = model
    return model


def get_embedding_dimension(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
) -> int:
    """
    Return how many numbers this model produces per text.

    For all-MiniLM-L6-v2 the answer is 384. The search index built later
    has to be told this number up front, so it is worth being able to ask
    the model rather than hard-coding it.
    """
    model = load_embedding_model(model_name, device)

    # sentence-transformers 6.x renamed this method. We prefer the new name
    # and fall back to the old one so this works on either version.
    if hasattr(model, "get_embedding_dimension"):
        return int(model.get_embedding_dimension())
    return int(model.get_sentence_embedding_dimension())


# ----------------------------------------------------------------------
# Internal helper
# ----------------------------------------------------------------------

def _validate_chunks(chunks: list[dict]) -> None:
    """
    Check the input really is chunk data before we spend time embedding it.

    Loading the model and running it is slow, so a clear error now is much
    kinder than a confusing one several seconds in.
    """
    required = ("chunk_id", "page_number", "text")

    for position, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise EmbeddingError(
                f"Item {position} of 'chunks' is a {type(chunk).__name__}, "
                "but a dictionary was expected.\n"
                "Pass the list returned by text_chunker.chunk_pages()."
            )

        missing = [key for key in required if key not in chunk]
        if missing:
            raise EmbeddingError(
                f"Item {position} of 'chunks' is missing the key(s): "
                f"{', '.join(missing)}.\n"
                "Each chunk must look like "
                '{"chunk_id": "p1-c0", "page_number": 1, "text": "..."}.'
            )

        if not isinstance(chunk["text"], str):
            raise EmbeddingError(
                f"Chunk '{chunk['chunk_id']}' has a "
                f"{type(chunk['text']).__name__} in its 'text' field, "
                "but a string was expected."
            )


# ----------------------------------------------------------------------
# Embedding chunks
# ----------------------------------------------------------------------

def embed_chunks(
    chunks: list[dict] | None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    normalize: bool = DEFAULT_NORMALIZE,
    show_progress: bool = False,
) -> list[dict]:
    """
    Convert every chunk's text into an embedding vector.

    The chunk's identity travels with its vector. That matters: a bare list
    of numbers is useless on its own, because after searching we need to
    know *which* chunk matched and *which page* it came from.

    Parameters
    ----------
    chunks:
        The list returned by `text_chunker.chunk_pages()`. Each item must
        have "chunk_id", "page_number" and "text". An empty list or None
        returns an empty list without loading the model at all.
    model_name:
        Which model to use. Defaults to "all-MiniLM-L6-v2".
    device:
        None (let the library choose), "cpu", or "mps" for the Apple
        Silicon GPU.
    batch_size:
        How many texts to embed at once. Default 32.
    normalize:
        Scale each vector to length 1. Default True - see DEFAULT_NORMALIZE.
    show_progress:
        Print a progress bar. Useful for long documents, noisy for short
        ones, so it is off by default.

    Returns
    -------
    A list of dictionaries, in the same order as the input::

        {
            "chunk_id": "p1-c0",       # carried over unchanged
            "chunk_index": 0,          # carried over if present
            "page_number": 1,          # carried over unchanged
            "text": "Alice was ...",   # carried over unchanged
            "embedding": array([...], dtype=float32),   # 384 numbers
            "embedding_dim": 384,
        }

    Raises
    ------
    EmbeddingError
        If the model cannot be loaded, or a chunk is the wrong shape.
    """
    # Nothing to do - and importantly, do not load a 90 MB model to
    # discover that.
    if not chunks:
        return []

    _validate_chunks(chunks)

    model = load_embedding_model(model_name, device)

    # Embed all texts in one call rather than looping one at a time. The
    # model processes a batch far faster than the same texts individually.
    texts = [chunk["text"] for chunk in chunks]

    try:
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
    except Exception as exc:
        raise EmbeddingError(
            "Something went wrong while creating embeddings.\n"
            f"Technical detail: {type(exc).__name__}: {exc}"
        ) from None

    # float32 instead of float64: half the memory, no meaningful loss of
    # accuracy, and it is the format the search index will expect later.
    vectors = np.asarray(vectors, dtype=np.float32)

    embedded: list[dict] = []
    for chunk, vector in zip(chunks, vectors):
        record = {
            "chunk_id": chunk["chunk_id"],
            "page_number": chunk["page_number"],
            "text": chunk["text"],
            "embedding": vector,
            "embedding_dim": int(vector.shape[0]),
        }
        # Keep chunk_index if the chunker provided it - it is how we will
        # map a search result row back to its chunk.
        if "chunk_index" in chunk:
            record["chunk_index"] = chunk["chunk_index"]
        embedded.append(record)

    return embedded


def embed_query(
    question: str,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    normalize: bool = DEFAULT_NORMALIZE,
) -> np.ndarray:
    """
    Turn a single question into one embedding vector.

    The user's question must be embedded by the *same model* as the chunks,
    otherwise the numbers are not comparable - like measuring one thing in
    centimetres and the other in inches.

    Parameters
    ----------
    question:
        The text to embed.
    model_name, device, normalize:
        Same meaning as in `embed_chunks`. Leave them at their defaults so
        the question is embedded exactly like the chunks were.

    Returns
    -------
    A 1-dimensional numpy array of float32 - 384 numbers for the default
    model.

    Raises
    ------
    EmbeddingError
        If the question is empty, or the model cannot be loaded.
    """
    if not isinstance(question, str) or not question.strip():
        raise EmbeddingError(
            "The question is empty. Please provide some text to embed."
        )

    model = load_embedding_model(model_name, device)

    try:
        vector = model.encode(
            question,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise EmbeddingError(
            "Something went wrong while embedding the question.\n"
            f"Technical detail: {type(exc).__name__}: {exc}"
        ) from None

    return np.asarray(vector, dtype=np.float32)


# ----------------------------------------------------------------------
# Convenience helpers
# ----------------------------------------------------------------------

def embeddings_to_matrix(embedded_chunks: list[dict]) -> np.ndarray:
    """
    Stack the individual vectors into one 2-dimensional array.

    Shape is (number of chunks, embedding dimension) - for example
    (23, 384). The search index built in the next step wants all the
    vectors together in one block like this, rather than one at a time.

    An empty input returns an empty array with shape (0, 0).
    """
    if not embedded_chunks:
        return np.zeros((0, 0), dtype=np.float32)

    return np.vstack(
        [chunk["embedding"] for chunk in embedded_chunks]
    ).astype(np.float32)


def summarise_embeddings(embedded_chunks: list[dict]) -> dict:
    """
    Describe a set of embeddings without printing hundreds of numbers.

    Useful for a quick sanity check. If `normalize` was True, every vector
    length should be 1.0 (give or take a rounding error).

    Returns
    -------
    A dictionary like::

        {
            "total_embeddings": 23,
            "embedding_dim": 384,
            "pages_covered": 11,
            "mean_vector_length": 1.0,
            "all_same_dimension": True,
        }

    An empty input returns the same keys with zero/neutral values.
    """
    if not embedded_chunks:
        return {
            "total_embeddings": 0,
            "embedding_dim": 0,
            "pages_covered": 0,
            "mean_vector_length": 0.0,
            "all_same_dimension": True,
        }

    dims = {chunk["embedding_dim"] for chunk in embedded_chunks}
    lengths = [
        float(np.linalg.norm(chunk["embedding"])) for chunk in embedded_chunks
    ]

    return {
        "total_embeddings": len(embedded_chunks),
        "embedding_dim": embedded_chunks[0]["embedding_dim"],
        "pages_covered": len({c["page_number"] for c in embedded_chunks}),
        "mean_vector_length": round(sum(lengths) / len(lengths), 6),
        "all_same_dimension": len(dims) == 1,
    }


# ----------------------------------------------------------------------
# Manual test runner
#
# Only runs when you execute this file directly.
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pdf_processor import PdfProcessingError, extract_text_from_pdf
    from text_chunker import chunk_pages

    if len(sys.argv) < 2:
        print("Usage: python src/embedding_manager.py <path-to-pdf>")
        print("Example: python src/embedding_manager.py sample_pdfs/sample.pdf")
        sys.exit(1)

    try:
        pages = extract_text_from_pdf(sys.argv[1])
    except PdfProcessingError as error:
        print(f"\nCould not process that PDF:\n{error}\n")
        sys.exit(1)

    chunks = chunk_pages(pages)
    print(f"\nPages read     : {len(pages)}")
    print(f"Chunks to embed: {len(chunks)}")

    if not chunks:
        print("\nNo chunks to embed - the PDF appears to have no text.")
        sys.exit(0)

    print("\nLoading the model (first run downloads about 90 MB)...")
    started = time.perf_counter()
    try:
        dimension = get_embedding_dimension()
        embedded = embed_chunks(chunks)
    except EmbeddingError as error:
        print(f"\nEmbedding failed:\n{error}\n")
        sys.exit(1)
    elapsed = time.perf_counter() - started

    stats = summarise_embeddings(embedded)
    matrix = embeddings_to_matrix(embedded)

    print(f"Done in {elapsed:.2f} seconds.\n")
    print(f"Embeddings created : {stats['total_embeddings']}")
    print(f"Numbers per vector : {stats['embedding_dim']} (model reports {dimension})")
    print(f"Pages covered      : {stats['pages_covered']}")
    print(f"Mean vector length : {stats['mean_vector_length']} (1.0 means normalised)")
    print(f"Matrix shape       : {matrix.shape}")

    first = embedded[0]
    print(f"\n--- first embedded chunk ---")
    print(f"  chunk_id    : {first['chunk_id']}")
    print(f"  page_number : {first['page_number']}")
    print(f"  text        : {first['text'][:70]}...")
    print(f"  embedding   : [{first['embedding'][0]:.4f}, "
          f"{first['embedding'][1]:.4f}, {first['embedding'][2]:.4f}, "
          f"... ] ({first['embedding_dim']} numbers)")
    print()
