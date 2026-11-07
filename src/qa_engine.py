"""
qa_engine.py
============
Step 5 of the "Chat With Your PDF" backend: turning retrieved text into an
answer, using Google's Gemini model.

WHERE THIS SITS IN THE PIPELINE
-------------------------------
    pdf_processor      PDF file      -> page text
    text_chunker       page text     -> overlapping chunks
    embedding_manager  chunks        -> 384-number vectors
    vector_store       vectors       -> "which chunks match this question?"
    qa_engine (here)   those chunks  -> a written answer, with page numbers

THE IDEA: RETRIEVAL-AUGMENTED GENERATION (RAG)
----------------------------------------------
A language model knows a great deal about the world but nothing about YOUR
PDF. So we do not ask it to recall - we *give* it the relevant passages and
ask it to answer using only those. That is the "augmented" part of RAG.

Two rules make the answers trustworthy:

1. Answer only from the supplied context. If the passages do not contain
   the answer, say so rather than inventing one.
2. Cite the page numbers, so the user can check for themselves.

Both are enforced by the system instruction in SYSTEM_INSTRUCTION below,
and reinforced by how the context is formatted.

API KEY HANDLING
----------------
The key is read from the project's .env file with python-dotenv. It is
never hard-coded, never printed, never logged, and never included in an
error message - see `_redact()`, which strips it from any text on its way
out, just in case a library ever echoes it back.

Create your .env from the template and paste your key there::

    cp .env.example .env
    # then edit .env and set GEMINI_API_KEY=your-key

A free key comes from https://aistudio.google.com/apikey

Run this file directly to try it out:
    python src/qa_engine.py sample_pdfs/sample.pdf "your question here"
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------

# Which Gemini model to use. Override it without touching this file by
# adding a line to your .env:   GEMINI_MODEL=some-other-model
# Use list_available_models() to see what your key can access.
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# How many chunks to retrieve and send as context.
DEFAULT_TOP_K = 3

# Low temperature keeps answers factual and close to the context. Higher
# values make the model more creative, which is the opposite of what we
# want when quoting a document.
DEFAULT_TEMPERATURE = 0.2

# Upper limit on the answer length, in tokens.
DEFAULT_MAX_OUTPUT_TOKENS = 1024

# If even the best-matching chunk scores below this cosine similarity, the
# question is probably about something the PDF does not cover. We then say
# so directly instead of spending an API call. Deliberately permissive -
# it only catches clearly unrelated questions.
MIN_USEFUL_SCORE = 0.15

# The sentence we use when the PDF cannot answer the question. Kept as a
# constant so the model's instruction and our own fallback match exactly.
NOT_ENOUGH_INFO = (
    "The PDF does not contain enough information to answer that question."
)

SYSTEM_INSTRUCTION = f"""\
You are a careful assistant that answers questions about a single PDF
document. You will be given numbered excerpts from that PDF, each labelled
with the page it came from.

Follow these rules without exception:

1. Answer ONLY using the supplied excerpts. Do not use outside knowledge,
   and do not guess or fill in gaps.
2. If the excerpts do not contain enough information to answer, reply with
   exactly this sentence and nothing more:
   "{NOT_ENOUGH_INFO}"
3. When you do answer, cite the page number(s) you used, in the form
   (page 2) or (pages 2, 3). Place the citation next to the claim it
   supports.
4. Do not mention "excerpts", "chunks", "context" or "the provided text"
   in your answer. Write naturally, as though you had read the document.
5. Be concise. Two or three sentences is usually plenty.
"""


class QAEngineError(Exception):
    """
    Raised when a question cannot be answered.

    Consistent with the other modules (PdfProcessingError, ChunkingError,
    EmbeddingError, VectorStoreError): one error type, with a message
    written for a person rather than a programmer.
    """


# The client is reusable, so build it once and keep it.
_CLIENT_CACHE: dict[str, object] = {}


# ----------------------------------------------------------------------
# API key handling
# ----------------------------------------------------------------------

def _project_root() -> Path:
    """Return the project folder (the parent of this file's src/ folder)."""
    return Path(__file__).resolve().parent.parent


def load_api_key() -> str:
    """
    Read GEMINI_API_KEY from the project's .env file.

    The returned string is handed straight to the Gemini client and must
    never be printed, logged or shown in a user interface.

    Returns
    -------
    The API key as a string.

    Raises
    ------
    QAEngineError
        If no .env file exists, or the key is missing or blank. The message
        explains how to fix it without ever revealing any value.
    """
    env_path = _project_root() / ".env"

    if not env_path.exists():
        raise QAEngineError(
            "No .env file was found in the project folder.\n"
            "Create one from the template and add your key:\n"
            "  cp .env.example .env\n"
            "Then edit .env and set GEMINI_API_KEY=your-key\n"
            "A free key is available at https://aistudio.google.com/apikey"
        )

    # override=True so that editing .env takes effect without restarting
    # a long-running session.
    load_dotenv(env_path, override=True)

    key = os.environ.get("GEMINI_API_KEY", "")
    if not key.strip():
        raise QAEngineError(
            "GEMINI_API_KEY is empty in your .env file.\n"
            "Open .env and put your key after the '=' sign, like:\n"
            "  GEMINI_API_KEY=your-key-here\n"
            "(no quotes, no spaces around the '=')\n"
            "A free key is available at https://aistudio.google.com/apikey"
        )

    return key.strip()


def _redact(text: str, key: str) -> str:
    """
    Remove the API key from a piece of text, should it ever appear.

    Library error messages are not supposed to echo credentials, but this
    costs nothing and means a stack trace can never leak the key.
    """
    if not key:
        return text
    cleaned = text.replace(key, "[REDACTED]")
    # Also redact a long prefix, in case only part of the key is echoed.
    if len(key) > 12:
        cleaned = cleaned.replace(key[:12], "[REDACTED]")
    return cleaned


def get_client(api_key: str | None = None):
    """
    Create (or reuse) the Gemini client.

    Parameters
    ----------
    api_key:
        Normally leave this as None so the key is read from .env. The
        parameter exists for testing.

    Returns
    -------
    A google.genai.Client.

    Raises
    ------
    QAEngineError
        If the SDK is missing or the client cannot be created.
    """
    key = api_key or load_api_key()

    # Cache on a fingerprint, never the key itself.
    cache_key = f"len{len(key)}:{hash(key)}"
    if cache_key in _CLIENT_CACHE:
        return _CLIENT_CACHE[cache_key]

    try:
        from google import genai
    except ImportError as exc:
        raise QAEngineError(
            "The 'google-genai' package is not available.\n"
            "Activate your virtual environment and install it with:\n"
            '  python -m pip install "google-genai<3.0.0"'
        ) from exc

    try:
        client = genai.Client(api_key=key)
    except Exception as exc:
        raise QAEngineError(
            "Could not create the Gemini client.\n"
            f"Technical detail: {type(exc).__name__}: "
            f"{_redact(str(exc), key)}"
        ) from None

    _CLIENT_CACHE[cache_key] = client
    return client


def list_available_models() -> list[str]:
    """
    Ask the API which models your key can use.

    Handy when the configured model name is rejected: model names change
    over time, and this tells you the current ones rather than guessing.

    Returns
    -------
    A sorted list of model names.

    Raises
    ------
    QAEngineError
        If the key is missing or invalid, or the request fails.
    """
    key = load_api_key()
    client = get_client(key)

    try:
        names = []
        for model in client.models.list():
            actions = getattr(model, "supported_actions", None) or []
            if not actions or "generateContent" in actions:
                names.append(model.name)
        return sorted(names)
    except Exception as exc:
        raise QAEngineError(
            _friendly_api_message(exc, key, DEFAULT_GEMINI_MODEL)
        ) from None


# ----------------------------------------------------------------------
# Building the prompt
# ----------------------------------------------------------------------

def build_context(matches: list[dict]) -> str:
    """
    Format retrieved chunks into labelled, numbered excerpts.

    Labelling each excerpt with its page number is what makes citation
    possible: the model can only cite a page if it was told which page the
    words came from.

    Parameters
    ----------
    matches:
        Results from `vector_store.search()` - each needs "page_number"
        and "text".

    Returns
    -------
    One string, ready to put in the prompt.
    """
    blocks = []
    for position, match in enumerate(matches, start=1):
        text = " ".join(match["text"].split())
        blocks.append(
            f"--- Excerpt {position} (page {match['page_number']}) ---\n"
            f"{text}"
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, matches: list[dict]) -> str:
    """
    Assemble the full message sent to Gemini.

    The question is placed AFTER the excerpts. Models attend well to the
    end of a prompt, so putting the question last keeps it prominent.
    """
    return (
        "Here are the relevant excerpts from the PDF:\n\n"
        f"{build_context(matches)}\n\n"
        "---\n\n"
        f"Question: {question}\n\n"
        "Answer using only the excerpts above, and cite the page "
        "number(s) you used."
    )


# ----------------------------------------------------------------------
# Error translation
# ----------------------------------------------------------------------

def _friendly_api_message(exc: Exception, key: str, model: str) -> str:
    """
    Turn a Gemini SDK exception into something a beginner can act on.

    The raw errors are long JSON blobs. This picks out the common causes
    and says what to do about each, with the key redacted throughout.
    """
    detail = _redact(str(exc), key)
    lowered = detail.lower()
    name = type(exc).__name__

    if "api_key_invalid" in lowered or "api key not valid" in lowered:
        return (
            "Gemini rejected your API key.\n"
            "Open .env and check that GEMINI_API_KEY holds a valid key:\n"
            "  - it should be a single line, with no quotes or spaces\n"
            "  - get a fresh one at https://aistudio.google.com/apikey\n"
            "  - a Google AI Studio key is a short string beginning "
            "'AIza', not a URL or a JSON file\n"
            f"({name}: the key itself is never shown here.)"
        )

    if "permission" in lowered or "403" in lowered:
        return (
            "Your API key was recognised but is not allowed to use this "
            f"model ('{model}').\n"
            "Check that the Generative Language API is enabled for your "
            "project, or try a different model.\n"
            "Run list_available_models() to see what your key can use."
        )

    if "not found" in lowered or "404" in lowered:
        return (
            f"The model '{model}' was not found.\n"
            "Model names change over time. To see what your key can use:\n"
            "  python -c \"import sys; sys.path.insert(0,'src'); "
            "from qa_engine import list_available_models; "
            "print(list_available_models())\"\n"
            "Then set the one you want in .env:  GEMINI_MODEL=that-name"
        )

    if "quota" in lowered or "resource_exhausted" in lowered or "429" in lowered:
        return (
            "You have hit Gemini's rate limit or quota.\n"
            "Wait a minute and try again. The free tier allows a limited "
            "number of requests per minute."
        )

    if "deadline" in lowered or "timeout" in lowered:
        return (
            "The request to Gemini timed out.\n"
            "Check your internet connection and try again."
        )

    if name == "ServerError" or "503" in lowered or "500" in lowered:
        return (
            "Gemini's service returned a server error, which usually means "
            "a temporary problem on Google's side.\n"
            "Wait a moment and try again."
        )

    return (
        "The request to Gemini failed.\n"
        f"Technical detail: {name}: {detail[:400]}"
    )


# ----------------------------------------------------------------------
# The main entry point
# ----------------------------------------------------------------------

def answer_question(
    store: dict,
    question: str,
    top_k: int = DEFAULT_TOP_K,
    model: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    min_score: float = MIN_USEFUL_SCORE,
    api_key: str | None = None,
) -> dict:
    """
    Answer a question about the indexed PDF.

    Steps, in order:
      1. embed the question           (embedding_manager.embed_query)
      2. retrieve the best chunks     (vector_store.search)
      3. format them with page labels (build_prompt)
      4. ask Gemini, constrained to that context

    Parameters
    ----------
    store:
        A store built by `vector_store.build_vector_store()`.
    question:
        The user's question.
    top_k:
        How many chunks to retrieve and send. Default 3.
    model:
        Gemini model name. Defaults to DEFAULT_GEMINI_MODEL, which can be
        set in .env as GEMINI_MODEL.
    temperature:
        0.0 is most literal, 1.0 most creative. Default 0.2.
    max_output_tokens:
        Maximum answer length. Default 1024.
    min_score:
        If the best chunk scores below this, we answer "not enough
        information" without calling the API. Default 0.15.
    api_key:
        For testing only. Normally leave as None to read from .env.

    Returns
    -------
    A dictionary::

        {
            "answer": "Alice found a tiny golden key (page 2).",
            "sources": [2, 3],            # pages used, ascending
            "matches": [...],             # the raw search results
            "model": "gemini-2.5-flash",
            "used_api": True,             # False if answered locally
        }

    Raises
    ------
    QAEngineError
        If the question is empty, the store is unusable, the API key is
        missing or invalid, or the API call fails.
    """
    if not isinstance(question, str) or not question.strip():
        raise QAEngineError(
            "The question is empty. Please type a question to ask about "
            "the PDF."
        )

    question = question.strip()
    model_name = model or DEFAULT_GEMINI_MODEL

    # --- step 1 and 2: retrieve ---------------------------------------
    # Imported here rather than at module level so that this file can be
    # imported without starting up PyTorch.
    from embedding_manager import EmbeddingError, embed_query
    from vector_store import VectorStoreError, search

    try:
        query_vector = embed_query(question)
    except EmbeddingError as exc:
        raise QAEngineError(
            f"Could not understand the question.\n{exc}"
        ) from None

    try:
        matches = search(store, query_vector, top_k=top_k)
    except VectorStoreError as exc:
        raise QAEngineError(
            f"Could not search the PDF.\n{exc}"
        ) from None

    # Requirement: handle empty retrieval results.
    if not matches:
        raise QAEngineError(
            "No text was retrieved from the PDF, so there is nothing to "
            "answer from.\n"
            "Make sure a PDF has been loaded and indexed first."
        )

    pages_used = sorted({match["page_number"] for match in matches})

    # Requirement: say so when the context cannot support an answer.
    # Catching obviously-unrelated questions here saves a pointless call.
    best_score = max(match["score"] for match in matches)
    if best_score < min_score:
        return {
            "answer": NOT_ENOUGH_INFO,
            "sources": [],
            "matches": matches,
            "model": model_name,
            "used_api": False,
        }

    # --- step 3 and 4: ask Gemini -------------------------------------
    key = api_key or load_api_key()
    client = get_client(key)

    try:
        from google.genai import types
    except ImportError as exc:
        raise QAEngineError(
            "The 'google-genai' package is incomplete or not installed.\n"
            'Reinstall it with: python -m pip install "google-genai<3.0.0"'
        ) from exc

    prompt = build_prompt(question, matches)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
    except Exception as exc:
        raise QAEngineError(
            _friendly_api_message(exc, key, model_name)
        ) from None

    answer = (getattr(response, "text", None) or "").strip()

    if not answer:
        raise QAEngineError(
            "Gemini returned an empty answer.\n"
            "This can happen if the response was blocked by a safety "
            "filter or cut off by max_output_tokens. Try rephrasing the "
            "question."
        )

    # If the model said it could not answer, report no sources - citing
    # pages for a non-answer would be misleading.
    declined = NOT_ENOUGH_INFO.lower().rstrip(".") in answer.lower()

    return {
        "answer": answer,
        "sources": [] if declined else pages_used,
        "matches": matches,
        "model": model_name,
        "used_api": True,
    }


def build_store_from_pdf(pdf_path: str | Path, **chunk_kwargs) -> dict:
    """
    Convenience: run the whole pipeline and return a searchable store.

    Exists so a caller (and the test runner below) can go from a file path
    to something answerable in one line.

    Parameters
    ----------
    pdf_path:
        Path to a .pdf file.
    **chunk_kwargs:
        Passed to `text_chunker.chunk_pages()`, for example
        chunk_size_tokens=250.

    Returns
    -------
    A store, as returned by `vector_store.build_vector_store()`.

    Raises
    ------
    QAEngineError
        If the PDF cannot be read, has no text, or cannot be indexed.
    """
    from embedding_manager import EmbeddingError, embed_chunks
    from pdf_processor import PdfProcessingError, extract_text_from_pdf
    from text_chunker import ChunkingError, chunk_pages
    from vector_store import VectorStoreError, build_vector_store

    try:
        pages = extract_text_from_pdf(pdf_path)
    except PdfProcessingError as exc:
        raise QAEngineError(str(exc)) from None

    try:
        chunks = chunk_pages(pages, **chunk_kwargs)
    except ChunkingError as exc:
        raise QAEngineError(str(exc)) from None

    if not chunks:
        raise QAEngineError(
            "No text could be extracted from that PDF, so there is nothing "
            "to ask questions about.\n"
            "If it is a scanned document, it would need OCR first."
        )

    try:
        embedded = embed_chunks(chunks)
        return build_vector_store(embedded)
    except (EmbeddingError, VectorStoreError) as exc:
        raise QAEngineError(str(exc)) from None


# ----------------------------------------------------------------------
# Manual test runner
#
# Only runs when you execute this file directly. It prints the question,
# the answer and the page numbers - never the API key.
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample_pdfs/sample.pdf"
    question = (
        " ".join(sys.argv[2:])
        if len(sys.argv) > 2
        else "What did Alice find on the little glass table?"
    )

    print(f"\nPDF      : {pdf_path}")
    print(f"Question : {question}")

    # Report only whether a key is configured - never the key itself.
    try:
        _key = load_api_key()
        print(f"API key  : configured ({len(_key)} characters, value hidden)")
    except QAEngineError as error:
        print(f"\nAPI key problem:\n{error}\n")
        sys.exit(1)

    print("\nIndexing the PDF...")
    try:
        store = build_store_from_pdf(pdf_path)
    except QAEngineError as error:
        print(f"\n{error}\n")
        sys.exit(1)
    print(f"  {store['size']} chunks indexed, {store['dimension']} dimensions")

    print(f"\nAsking Gemini ({DEFAULT_GEMINI_MODEL})...")
    try:
        result = answer_question(store, question, top_k=DEFAULT_TOP_K)
    except QAEngineError as error:
        print(f"\nCould not answer:\n{error}\n")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result["answer"])
    print()
    print(f"Source page(s) : {result['sources'] or 'none'}")
    print(f"Model          : {result['model']}")
    print(f"Called the API : {result['used_api']}")
    print("\nRetrieved excerpts used as context:")
    for match in result["matches"]:
        preview = " ".join(match["text"].split())[:66]
        print(f"  {match['rank']}. score {match['score']:+.4f}  "
              f"{match['chunk_id']:<8} page {match['page_number']:<3} "
              f"\"{preview}...\"")
    print()
