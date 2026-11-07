"""
pdf_processor.py
================
Step 1 of the "Chat With Your PDF" backend: turning a PDF file into text.

A PDF is not a text file. It is a page-layout format that describes where
each character should be drawn on a page. This module uses `pypdf` to walk
through a PDF page by page and reassemble that drawing information back
into readable strings.

Nothing in this module talks to an AI model, builds embeddings, or draws a
user interface. It does exactly one job: PDF file in, text out.

Public functions
----------------
extract_text_from_pdf(pdf_path)  -> list[dict]   main entry point
get_pdf_info(pdf_path)           -> dict         quick summary of a PDF
PdfProcessingError                               the one error type we raise

Run this file directly to try it out:
    python src/pdf_processor.py sample_pdfs/your_file.pdf
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import (
    DependencyError,
    EmptyFileError,
    FileNotDecryptedError,
    PdfReadError,
    PyPdfError,
    WrongPasswordError,
)

# How many characters of text a page must have before we consider it
# "has real text". Scanned pages often yield a stray space or newline.
MIN_CHARS_FOR_REAL_TEXT = 3


class PdfProcessingError(Exception):
    """
    Raised when a PDF cannot be read.

    We define our own error type so that calling code (later: Streamlit)
    can catch exactly one thing and show the message straight to the user.
    Every message raised here is written to be understandable by a person,
    not a programmer.
    """


# ----------------------------------------------------------------------
# Internal helpers (the leading underscore means "private to this file")
# ----------------------------------------------------------------------

def _validate_path(pdf_path: str | Path) -> Path:
    """
    Check the file exists and looks like a PDF, before pypdf ever sees it.

    Catching these problems early gives a much friendlier message than
    letting pypdf fail deep inside its parser.
    """
    path = Path(pdf_path).expanduser()

    if not path.exists():
        raise PdfProcessingError(
            f"Could not find a file at: {path}\n"
            "Please check the file name and the folder path are correct."
        )

    if path.is_dir():
        raise PdfProcessingError(
            f"That path is a folder, not a file: {path}\n"
            "Please point to a single .pdf file instead."
        )

    if path.suffix.lower() != ".pdf":
        raise PdfProcessingError(
            f"This file does not look like a PDF: {path.name}\n"
            f"Expected a file ending in '.pdf' but found '{path.suffix}'."
        )

    if path.stat().st_size == 0:
        raise PdfProcessingError(
            f"The file is empty (0 bytes): {path.name}\n"
            "The PDF may have failed to download or copy correctly."
        )

    return path


def _open_reader(path: Path) -> PdfReader:
    """
    Open the PDF and translate pypdf's technical errors into plain English.

    strict=False tells pypdf to tolerate small spec violations instead of
    refusing to open the file. Many real-world PDFs are slightly malformed,
    and being lenient here means fewer files are rejected unnecessarily.
    """
    try:
        reader = PdfReader(path, strict=False)
    except EmptyFileError:
        raise PdfProcessingError(
            f"'{path.name}' contains no PDF data at all.\n"
            "The file is likely incomplete or corrupted."
        ) from None
    except (PdfReadError, OSError) as exc:
        raise PdfProcessingError(
            f"'{path.name}' could not be opened as a PDF.\n"
            "The file may be corrupted, or it may be another file type "
            "that was renamed to end in '.pdf'.\n"
            f"Technical detail: {exc}"
        ) from None

    # A password-protected PDF opens fine but refuses to give up its pages.
    if reader.is_encrypted:
        try:
            # An empty password unlocks PDFs that are only "owner protected"
            # (restricted printing/copying but readable). Returns 0 on failure.
            if reader.decrypt("") == 0:
                raise PdfProcessingError(
                    f"'{path.name}' is password-protected.\n"
                    "Please open it in a PDF reader, save an unlocked copy, "
                    "and use that copy instead."
                )
        except (WrongPasswordError, FileNotDecryptedError):
            raise PdfProcessingError(
                f"'{path.name}' is password-protected.\n"
                "Please open it in a PDF reader, save an unlocked copy, "
                "and use that copy instead."
            ) from None
        except DependencyError:
            raise PdfProcessingError(
                f"'{path.name}' uses strong encryption that needs an extra "
                "package to read.\n"
                "Install it with:  python -m pip install pycryptodome"
            ) from None

    return reader


def _tidy(text: str) -> str:
    """
    Light cleanup of extracted text.

    PDF extraction often produces trailing spaces and runs of blank lines.
    We trim each line and collapse 3+ blank lines down to one, which makes
    the text easier to read and slightly cheaper to send to a model later.
    We deliberately do NOT reflow paragraphs - that would risk changing
    meaning, and the exact wording matters for answering questions.
    """
    lines = [line.rstrip() for line in text.splitlines()]

    tidied: list[str] = []
    blank_run = 0
    for line in lines:
        if line:
            blank_run = 0
            tidied.append(line)
        else:
            blank_run += 1
            if blank_run == 1:          # keep the first blank line only
                tidied.append("")

    return "\n".join(tidied).strip()


# ----------------------------------------------------------------------
# Public functions
# ----------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str | Path) -> list[dict]:
    """
    Extract the text of a PDF, one entry per page.

    Parameters
    ----------
    pdf_path:
        Path to a .pdf file. Accepts a string or a pathlib.Path.

    Returns
    -------
    A list of dictionaries, in page order. Each dictionary looks like::

        {
            "page_number": 1,          # 1-based: page 1 is the first page
            "text": "Chapter one ...", # extracted text ("" if none found)
            "char_count": 1432,        # length of "text", for quick checks
        }

    Page numbering is 1-based so that it matches what a human sees in a
    PDF reader. (Python lists start at 0, so page_number is index + 1.)

    Raises
    ------
    PdfProcessingError
        If the file is missing, empty, not a PDF, password-protected,
        or too corrupted to read.
    """
    path = _validate_path(pdf_path)
    reader = _open_reader(path)

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise PdfProcessingError(
            f"'{path.name}' opened successfully but contains no pages."
        )

    pages: list[dict] = []

    # enumerate(..., start=1) gives us 1-based page numbers directly.
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:
            # One unreadable page should not throw away the whole document.
            # We record the page with empty text and keep going.
            raw_text = ""
            print(
                f"  [warning] Page {page_number} of '{path.name}' could not "
                f"be read and was skipped ({type(exc).__name__}: {exc})"
            )

        text = _tidy(raw_text)
        pages.append(
            {
                "page_number": page_number,
                "text": text,
                "char_count": len(text),
            }
        )

    return pages


def get_pdf_info(pdf_path: str | Path) -> dict:
    """
    Summarise a PDF without returning all of its text.

    Useful for showing the user what was loaded, and for detecting scanned
    documents before we waste time sending them to an AI model.

    Returns
    -------
    A dictionary like::

        {
            "file_name": "report.pdf",
            "total_pages": 12,
            "pages_with_text": 12,
            "total_characters": 24680,
            "looks_scanned": False,   # True = probably images, needs OCR
        }

    Raises
    ------
    PdfProcessingError
        Same conditions as extract_text_from_pdf.
    """
    path = _validate_path(pdf_path)
    pages = extract_text_from_pdf(path)

    pages_with_text = sum(
        1 for page in pages if page["char_count"] >= MIN_CHARS_FOR_REAL_TEXT
    )
    total_characters = sum(page["char_count"] for page in pages)

    return {
        "file_name": path.name,
        "total_pages": len(pages),
        "pages_with_text": pages_with_text,
        "total_characters": total_characters,
        # A PDF of scanned images has pages but almost no extractable text.
        "looks_scanned": pages_with_text == 0,
    }


# ----------------------------------------------------------------------
# Manual test runner
#
# This block only runs when you execute the file directly. It does not run
# when another file imports this module - that is what __name__ checks.
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/pdf_processor.py <path-to-pdf>")
        print("Example: python src/pdf_processor.py sample_pdfs/example.pdf")
        sys.exit(1)

    target = sys.argv[1]

    try:
        info = get_pdf_info(target)
        extracted = extract_text_from_pdf(target)
    except PdfProcessingError as error:
        print(f"\nCould not process that PDF:\n{error}\n")
        sys.exit(1)

    print(f"\nFile              : {info['file_name']}")
    print(f"Total pages       : {info['total_pages']}")
    print(f"Pages with text   : {info['pages_with_text']}")
    print(f"Total characters  : {info['total_characters']:,}")

    if info["looks_scanned"]:
        print(
            "\nNo text could be extracted from any page.\n"
            "This is almost certainly a scanned PDF (pictures of pages "
            "rather than real text).\nReading it would require OCR, which "
            "this project does not do yet."
        )
        sys.exit(0)

    print("\n--- preview: first 300 characters of each page ---")
    for page in extracted:
        preview = page["text"][:300].replace("\n", " ")
        suffix = "..." if page["char_count"] > 300 else ""
        print(f"\n[Page {page['page_number']}] ({page['char_count']} chars)")
        print(f"  {preview}{suffix}" if preview else "  (no text on this page)")
    print()
