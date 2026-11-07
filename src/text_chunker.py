"""
text_chunker.py
===============
Step 2 of the "Chat With Your PDF" backend: cutting page text into chunks.

WHY CHUNKING IS NEEDED
----------------------
After `pdf_processor` gives us the text of each page, we cannot simply hand
a whole document to an AI model and ask questions about it. Two reasons:

1. Retrieval accuracy. To answer "what did Alice find on the glass table?"
   we want to find the *specific paragraph* that answers it, not the whole
   book. Searching works far better over small passages than huge ones.
2. Size limits. Models and embedding functions accept a limited amount of
   text at once.

So we slice the text into overlapping pieces ("chunks"), and later we will
turn each chunk into a vector and search over them.

This module imports nothing except the standard library. It does not touch
pypdf, embeddings, FAISS, Gemini, or Streamlit. Text in, chunks out.

INPUT CONTRACT (produced by src/pdf_processor.py)
-------------------------------------------------
A list of dictionaries, one per page::

    [
        {"page_number": 1, "text": "Alice was beginning...", "char_count": 3006},
        {"page_number": 2, "text": "be four thousand...",    "char_count": 3673},
    ]

OUTPUT (produced by this module)
--------------------------------
A list of dictionaries, one per chunk - see `chunk_pages` for the shape.

Run this file directly to try it out:
    python src/text_chunker.py sample_pdfs/sample.pdf
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# Tunable settings
# ----------------------------------------------------------------------

# Target chunk size and overlap, measured in *tokens*.
DEFAULT_CHUNK_SIZE_TOKENS = 250
DEFAULT_OVERLAP_TOKENS = 50

# A "token" is roughly a word-piece: common words are one token, longer or
# unusual words split into several. We do not load a real tokenizer here
# (that would mean pulling in a model just to count words), so we estimate.
#
# The widely used rule of thumb for English prose is ~1.3 tokens per word.
# That makes 250 tokens land at about 192 words. "Approximately" is fine:
# chunk size only needs to be in the right ballpark, not exact.
AVERAGE_TOKENS_PER_WORD = 1.3

# A page needs at least this many non-whitespace characters to be chunked.
# Blank pages and scanned pages often yield a stray space or newline.
# This matches MIN_CHARS_FOR_REAL_TEXT in pdf_processor.py.
MIN_MEANINGFUL_CHARS = 3


class ChunkingError(Exception):
    """
    Raised when the input data or the settings do not make sense.

    As in pdf_processor, we use one custom error type so that calling code
    can catch exactly one thing and show the message to the user.
    """


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _meaningful_char_count(text: str) -> int:
    """
    Count characters that are not whitespace.

    "   \n  " has a length of 6 but zero meaningful characters, so plain
    len() would wrongly treat a blank page as having content.
    """
    return len("".join(text.split()))


def _tokens_to_words(tokens: int) -> int:
    """
    Convert a token count into an approximate word count.

    We slide our window over words rather than tokens because splitting on
    whitespace is simple, fast, and never cuts a word in half.
    """
    return max(1, round(tokens / AVERAGE_TOKENS_PER_WORD))


def _estimate_tokens(text: str) -> int:
    """Estimate how many tokens a piece of text is worth."""
    word_count = len(text.split())
    return round(word_count * AVERAGE_TOKENS_PER_WORD)


def _validate_settings(chunk_size_tokens: int, overlap_tokens: int) -> None:
    """
    Reject settings that would produce broken or infinite output.

    The important one is overlap >= chunk size: the window would never move
    forward (or would move backwards), so chunking would loop forever.
    """
    if chunk_size_tokens <= 0:
        raise ChunkingError(
            f"chunk_size_tokens must be greater than 0, got {chunk_size_tokens}."
        )

    if overlap_tokens < 0:
        raise ChunkingError(
            f"overlap_tokens cannot be negative, got {overlap_tokens}."
        )

    if overlap_tokens >= chunk_size_tokens:
        raise ChunkingError(
            f"overlap_tokens ({overlap_tokens}) must be smaller than "
            f"chunk_size_tokens ({chunk_size_tokens}).\n"
            "If the overlap were as large as the chunk, the chunks would "
            "never advance through the text."
        )


# ----------------------------------------------------------------------
# Public functions
# ----------------------------------------------------------------------

def split_text_into_chunks(
    text: str,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[str]:
    """
    Split one piece of text into overlapping chunks.

    This is the low-level worker: it knows nothing about pages. It takes a
    string and returns a list of strings.

    Parameters
    ----------
    text:
        The text to split.
    chunk_size_tokens:
        Approximate size of each chunk, in tokens. Default 250.
    overlap_tokens:
        Approximate number of tokens each chunk repeats from the previous
        chunk. Default 50. Must be smaller than chunk_size_tokens.

    Returns
    -------
    A list of chunk strings, in order. Returns an empty list if the text
    has no meaningful characters.

    Raises
    ------
    ChunkingError
        If the size/overlap settings are invalid.
    """
    _validate_settings(chunk_size_tokens, overlap_tokens)

    if _meaningful_char_count(text) < MIN_MEANINGFUL_CHARS:
        return []

    # Work in words: simple, and it never splits a word down the middle.
    words = text.split()
    if not words:
        return []

    words_per_chunk = _tokens_to_words(chunk_size_tokens)
    words_of_overlap = _tokens_to_words(overlap_tokens)

    # "step" is how far the window slides each time. Because step is
    # smaller than the window, consecutive windows share some words -
    # that shared part is the overlap.
    step = words_per_chunk - words_of_overlap

    chunks: list[str] = []
    start = 0

    while start < len(words):
        window = words[start:start + words_per_chunk]
        chunks.append(" ".join(window))

        # If this window reached the end of the text, stop. Without this
        # check the final short window would be emitted again as a chunk
        # that is entirely contained in the one before it.
        if start + words_per_chunk >= len(words):
            break

        start += step

    return chunks


def chunk_pages(
    pages: list[dict] | None,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[dict]:
    """
    Turn page data from pdf_processor into a flat list of chunks.

    Pages are chunked one at a time and never merged together, so every
    chunk belongs to exactly one page and can carry that page's number.
    This is what later lets the app say "this answer came from page 7".

    Parameters
    ----------
    pages:
        The list returned by `pdf_processor.extract_text_from_pdf()`.
        Each item must be a dict with "page_number" (int) and "text" (str).
        An empty list or None returns an empty list.
    chunk_size_tokens:
        Approximate size of each chunk, in tokens. Default 250.
    overlap_tokens:
        Approximate overlap between neighbouring chunks. Default 50.

    Returns
    -------
    A list of dictionaries, in document order. Each one looks like::

        {
            "chunk_id": "p1-c0",    # unique, readable: page 1, chunk 0
            "chunk_index": 0,       # position in this list (0-based)
            "page_number": 1,       # 1-based, copied from the page
            "text": "Alice was ...",
            "char_count": 1987,
            "approx_tokens": 498,
        }

    Pages with fewer than MIN_MEANINGFUL_CHARS (3) non-whitespace
    characters are skipped entirely and produce no chunks.

    Raises
    ------
    ChunkingError
        If the settings are invalid, or if a page is missing the
        "page_number" or "text" keys.
    """
    _validate_settings(chunk_size_tokens, overlap_tokens)

    # Graceful handling of "nothing to do".
    if not pages:
        return []

    chunks: list[dict] = []

    for position, page in enumerate(pages):
        # Fail loudly and clearly if the input is not the expected shape.
        if not isinstance(page, dict):
            raise ChunkingError(
                f"Item {position} of 'pages' is a {type(page).__name__}, "
                "but a dictionary was expected.\n"
                "Pass the list returned by "
                "pdf_processor.extract_text_from_pdf()."
            )

        missing = [key for key in ("page_number", "text") if key not in page]
        if missing:
            raise ChunkingError(
                f"Item {position} of 'pages' is missing the key(s): "
                f"{', '.join(missing)}.\n"
                "Each page must look like "
                '{"page_number": 1, "text": "..."}.'
            )

        page_number = page["page_number"]
        page_text = page["text"] or ""

        # Requirement: skip pages with fewer than 3 meaningful characters.
        if _meaningful_char_count(page_text) < MIN_MEANINGFUL_CHARS:
            continue

        page_chunks = split_text_into_chunks(
            page_text,
            chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
        )

        for chunk_number, chunk_text in enumerate(page_chunks):
            chunks.append(
                {
                    # Readable and unique: no page produces the same
                    # chunk_number twice.
                    "chunk_id": f"p{page_number}-c{chunk_number}",
                    # Position in the returned list. Later, FAISS will
                    # give us back row positions, and this makes the
                    # mapping back to a chunk obvious.
                    "chunk_index": len(chunks),
                    "page_number": page_number,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                    "approx_tokens": _estimate_tokens(chunk_text),
                }
            )

    return chunks


def summarise_chunks(chunks: list[dict]) -> dict:
    """
    Describe a list of chunks without printing all of their text.

    Handy for sanity-checking that chunk sizes look reasonable.

    Returns
    -------
    A dictionary like::

        {
            "total_chunks": 32,
            "pages_covered": 11,
            "total_characters": 34120,
            "smallest_chunk_tokens": 118,
            "largest_chunk_tokens": 501,
            "average_chunk_tokens": 437,
        }

    An empty input returns the same keys with zero values.
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "pages_covered": 0,
            "total_characters": 0,
            "smallest_chunk_tokens": 0,
            "largest_chunk_tokens": 0,
            "average_chunk_tokens": 0,
        }

    token_counts = [chunk["approx_tokens"] for chunk in chunks]

    return {
        "total_chunks": len(chunks),
        "pages_covered": len({chunk["page_number"] for chunk in chunks}),
        "total_characters": sum(chunk["char_count"] for chunk in chunks),
        "smallest_chunk_tokens": min(token_counts),
        "largest_chunk_tokens": max(token_counts),
        "average_chunk_tokens": round(sum(token_counts) / len(token_counts)),
    }


# ----------------------------------------------------------------------
# Manual test runner
#
# Only runs when you execute this file directly. pdf_processor is imported
# here, inside the block, so that importing this module elsewhere does not
# require pypdf.
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pdf_processor import PdfProcessingError, extract_text_from_pdf

    if len(sys.argv) < 2:
        print("Usage: python src/text_chunker.py <path-to-pdf>")
        print("Example: python src/text_chunker.py sample_pdfs/sample.pdf")
        sys.exit(1)

    try:
        pages = extract_text_from_pdf(sys.argv[1])
    except PdfProcessingError as error:
        print(f"\nCould not process that PDF:\n{error}\n")
        sys.exit(1)

    chunks = chunk_pages(pages)
    stats = summarise_chunks(chunks)

    print(f"\nPages read        : {len(pages)}")
    print(f"Chunks created    : {stats['total_chunks']}")
    print(f"Pages covered     : {stats['pages_covered']}")
    print(f"Chunk tokens      : min {stats['smallest_chunk_tokens']}, "
          f"avg {stats['average_chunk_tokens']}, "
          f"max {stats['largest_chunk_tokens']}")

    if not chunks:
        print("\nNo chunks were produced. The PDF appears to have no text.")
        sys.exit(0)

    print("\n--- first 2 chunks (first 200 characters each) ---")
    for chunk in chunks[:2]:
        print(f"\n[{chunk['chunk_id']}] page {chunk['page_number']}, "
              f"{chunk['approx_tokens']} tokens, index {chunk['chunk_index']}")
        print(f"  {chunk['text'][:200]}...")
    print()
