"""
app.py - Chat With Your PDF
===========================
The Streamlit user interface for the project.

This file is only the "front of house". All the real work lives in src/:

    src/pdf_processor.py      PDF file      -> page text
    src/text_chunker.py       page text     -> overlapping chunks
    src/embedding_manager.py  chunks        -> 384-number vectors
    src/vector_store.py       vectors       -> similarity search (FAISS)
    src/qa_engine.py          chunks        -> an answer from Gemini

Run it with:
    streamlit run app.py

IMPORTANT ORDERING NOTE
-----------------------
`load_dotenv()` is called BEFORE importing qa_engine. qa_engine decides its
default model name when it is first imported, by reading os.environ. If the
.env file has not been loaded by then, a GEMINI_MODEL setting in .env would
be ignored. Loading first makes .env work as expected.

The API key is read from .env by qa_engine. This file never reads, prints,
stores or displays it - the sidebar only reports whether one is configured.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# --- .env first, then our own modules (see note above) ----------------
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

sys.path.insert(0, str(PROJECT_ROOT / "src"))

import os  # noqa: E402  (imported after load_dotenv on purpose)

from embedding_manager import EmbeddingError, load_embedding_model  # noqa: E402
from pdf_processor import PdfProcessingError, get_pdf_info  # noqa: E402
from qa_engine import (  # noqa: E402
    QAEngineError,
    answer_question,
    build_store_from_pdf,
)

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------

# The model to use. Override it in .env with:
#   GEMINI_MODEL=models/gemini-3.5-flash
# We pass this explicitly to answer_question() rather than relying on the
# module default, so the app always uses a model we know works.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-3.5-flash")

# How many chunks to retrieve per question.
TOP_K = 3

PAGE_TITLE = "Chat With Your PDF"
PAGE_SUBTITLE = "Your AI-powered document assistant"


# ----------------------------------------------------------------------
# Page setup and styling
# ----------------------------------------------------------------------

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

# A small amount of CSS: tighten the top spacing, soften the subtitle, and
# give source citations a subtle pill look. Nothing structural.
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.5rem; max-width: 820px; }
      .app-subtitle {
          color: #6b7280;
          font-size: 1.02rem;
          margin-top: -0.6rem;
          margin-bottom: 1.6rem;
      }
      .source-pill {
          display: inline-block;
          background: rgba(125, 125, 125, 0.14);
          border-radius: 999px;
          padding: 0.12rem 0.62rem;
          margin-right: 0.35rem;
          font-size: 0.82rem;
      }
      .stChatMessage { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def warm_embedding_model():
    """
    Load the embedding model once per app session.

    Without caching, Streamlit would reload the ~90 MB model on every
    interaction, because it re-runs this whole script each time. The
    @st.cache_resource decorator makes the result persist instead.
    """
    return load_embedding_model()


# ----------------------------------------------------------------------
# Session state
#
# Streamlit re-runs this script top to bottom on every click or keypress.
# Anything that must survive a re-run (the index, the chat history) has to
# live in st.session_state.
# ----------------------------------------------------------------------

def init_state() -> None:
    """Create the session keys we rely on, once."""
    st.session_state.setdefault("store", None)       # the FAISS vector store
    st.session_state.setdefault("pdf_info", None)    # page counts etc.
    st.session_state.setdefault("file_signature", None)  # which PDF is loaded
    st.session_state.setdefault("messages", [])      # chat history


def file_signature(uploaded_file) -> str:
    """
    A short fingerprint of the uploaded bytes.

    Used to tell "the same file, re-sent on a script re-run" apart from
    "a genuinely new file the user just picked", so we only rebuild the
    index when the document actually changes.
    """
    data = uploaded_file.getvalue()
    return hashlib.sha1(data).hexdigest()[:16]


def reset_document_state() -> None:
    """Forget the current document and its conversation."""
    st.session_state.store = None
    st.session_state.pdf_info = None
    st.session_state.file_signature = None
    st.session_state.messages = []


# ----------------------------------------------------------------------
# Processing an upload
# ----------------------------------------------------------------------

def process_upload(uploaded_file) -> bool:
    """
    Read, chunk, embed and index an uploaded PDF.

    Streamlit hands us the file in memory, but our pdf_processor module
    works with a file path, so we write the bytes to a temporary file,
    process it, and let the temporary file be deleted afterwards. The
    index lives in memory, so nothing needs to stay on disk.

    Returns
    -------
    True if the document is ready to answer questions.
    """
    signature = file_signature(uploaded_file)

    # Already indexed this exact file - nothing to do.
    if st.session_state.store is not None and st.session_state.file_signature == signature:
        return True

    reset_document_state()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / uploaded_file.name

        try:
            temp_path.write_bytes(uploaded_file.getvalue())
        except OSError as error:
            st.error(f"Could not save the uploaded file: {error}")
            return False

        try:
            with st.spinner("Reading the PDF..."):
                info = get_pdf_info(temp_path)
        except PdfProcessingError as error:
            st.error(str(error))
            return False

        if info["looks_scanned"]:
            st.error(
                f"No text could be extracted from **{info['file_name']}**.\n\n"
                "This looks like a scanned PDF - pictures of pages rather "
                "than real text. Reading it would need OCR, which this app "
                "does not do yet. Please try a PDF with selectable text."
            )
            return False

        try:
            with st.spinner(
                f"Indexing {info['total_pages']} pages - "
                "this takes a few seconds..."
            ):
                store = build_store_from_pdf(temp_path)
        except (QAEngineError, EmbeddingError) as error:
            st.error(str(error))
            return False

    st.session_state.store = store
    st.session_state.pdf_info = info
    st.session_state.file_signature = signature
    return True


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

def render_sidebar(uploaded_file) -> None:
    """Show document details and app status."""
    with st.sidebar:
        st.subheader("Document")

        info = st.session_state.pdf_info
        if info is None:
            st.caption("No PDF loaded yet.")
        else:
            st.markdown(f"**{info['file_name']}**")
            left, right = st.columns(2)
            left.metric("Pages", info["total_pages"])
            right.metric("With text", info["pages_with_text"])

            store = st.session_state.store
            if store is not None:
                st.caption(
                    f"{store['size']} chunks indexed  ·  "
                    f"{info['total_characters']:,} characters"
                )

            if st.button("Clear document and chat", use_container_width=True):
                reset_document_state()
                st.rerun()

        st.divider()
        st.subheader("Settings")
        st.caption(f"**Model**  \n`{GEMINI_MODEL}`")
        st.caption(f"**Embeddings**  \n`all-MiniLM-L6-v2` (384-dim, local)")
        st.caption(f"**Retrieval**  \nTop {TOP_K} chunks per question")

        # Report only WHETHER a key is set. The value is never shown.
        key_present = bool(os.environ.get("GEMINI_API_KEY", "").strip())
        if key_present:
            st.success("API key configured", icon="✅")
        else:
            st.error("No API key found in .env", icon="⚠️")

        st.divider()
        st.caption(
            "Answers come only from the uploaded PDF. "
            "If the document does not cover a question, the app says so "
            "rather than guessing."
        )


# ----------------------------------------------------------------------
# Chat
# ----------------------------------------------------------------------

def render_sources(sources: list[int]) -> None:
    """Show the page numbers an answer was based on, as small pills."""
    if not sources:
        return
    label = "Source page" if len(sources) == 1 else "Source pages"
    pills = "".join(
        f"<span class='source-pill'>page {page}</span>" for page in sources
    )
    st.markdown(
        f"<div style='margin-top:0.5rem'>"
        f"<span style='color:#6b7280;font-size:0.82rem'>{label}: </span>"
        f"{pills}</div>",
        unsafe_allow_html=True,
    )


def render_excerpts(matches: list[dict]) -> None:
    """Show the retrieved chunks, collapsed by default."""
    if not matches:
        return
    with st.expander(f"Show the {len(matches)} excerpts used"):
        for match in matches:
            st.markdown(
                f"**Page {match['page_number']}** · "
                f"similarity {match['score']:+.3f} · `{match['chunk_id']}`"
            )
            st.caption(" ".join(match["text"].split()))
            if match is not matches[-1]:
                st.divider()


def render_history() -> None:
    """Redraw the whole conversation after a re-run."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))
                render_excerpts(message.get("matches", []))


def handle_question(question: str) -> None:
    """Answer one question and append both turns to the history."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the PDF and thinking..."):
                result = answer_question(
                    st.session_state.store,
                    question,
                    top_k=TOP_K,
                    model=GEMINI_MODEL,
                )
        except QAEngineError as error:
            st.error(str(error))
            # Keep the failure in the history so the user can see what
            # happened after a re-run, instead of it silently vanishing.
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"⚠️ {error}",
                    "sources": [],
                    "matches": [],
                }
            )
            return

        st.markdown(result["answer"])
        render_sources(result["sources"])
        render_excerpts(result["matches"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "matches": result["matches"],
        }
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    init_state()

    st.title(PAGE_TITLE)
    st.markdown(
        f"<div class='app-subtitle'>{PAGE_SUBTITLE}</div>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="A text-based PDF works best. Scanned documents need OCR.",
    )

    render_sidebar(uploaded_file)

    if uploaded_file is None:
        st.info(
            "Upload a PDF to get started. Once it is indexed you can ask "
            "questions about it, and every answer will cite the pages it "
            "came from.",
            icon="📄",
        )
        return

    # Warm the embedding model before the first question, so the delay
    # happens during indexing rather than mid-conversation.
    try:
        warm_embedding_model()
    except EmbeddingError as error:
        st.error(str(error))
        return

    if not process_upload(uploaded_file):
        return

    info = st.session_state.pdf_info
    st.success(
        f"**{info['file_name']}** is ready - "
        f"{info['total_pages']} pages, "
        f"{st.session_state.store['size']} chunks indexed.",
        icon="✅",
    )

    render_history()

    question = st.chat_input("Ask a question about this PDF...")
    if question and question.strip():
        handle_question(question.strip())


if __name__ == "__main__":
    main()
