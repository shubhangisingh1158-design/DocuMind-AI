# 📄 DocuMind AI — Intelligent PDF Assistant

**Ask questions about any PDF and get answers with page citations.**

A beginner-friendly Retrieval-Augmented Generation (RAG) application built with
Python and Streamlit. Upload a document, ask a question in plain English, and
receive an answer drawn **only** from that document — with the page numbers it
came from.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.67-FF4B4B?logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/Google%20Gemini-API-4285F4?logo=google&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-vector%20search-0467DF)


---

## 📖 Overview

Large language models know a great deal about the world — but nothing about
*your* documents. DocuMind AI closes that gap.

Instead of asking the model to recall information, the app **retrieves** the
most relevant passages from your PDF and gives them to Gemini as context, with
strict instructions to answer only from what it was shown. This is the pattern
known as **RAG (Retrieval-Augmented Generation)**, and it produces two things
that matter:

- **Grounded answers.** The model quotes your document, not the internet.
- **Verifiable citations.** Every answer names the pages it used, so you can
  check it yourself.

If the document genuinely does not contain the answer, the app says so plainly
rather than inventing one.

### Example

> **Question:** What did Alice find on the little glass table?
>
> **Answer:** Alice initially found a tiny golden key on the glass table
> *(page 2)*. When she returned to the table later, she also found a little
> bottle with a paper label around its neck with the words "DRINK ME" printed
> on it *(page 3)*.
>
> **Source pages:** `2` `3`

---

## ✨ Features

| | Feature |
|---|---|
| 📤 | **Drag-and-drop PDF upload** — any text-based PDF |
| 🔍 | **Semantic search** — finds meaning, not just keywords ("car" matches "automobile") |
| 💬 | **Chat interface** — ask follow-up questions, full history preserved |
| 📎 | **Page citations** — every answer shows the pages it came from |
| 🔬 | **Transparent retrieval** — expand any answer to see the exact excerpts used, with similarity scores |
| 🚫 | **No hallucination** — says "not enough information" instead of guessing |
| 🏠 | **Local embeddings** — text never leaves your machine during indexing |
| ⚡ | **Apple Silicon native** — runs on the M-series GPU via PyTorch MPS |
| 🛡️ | **Friendly error messages** — corrupted files, scanned PDFs, missing API keys all explained in plain English |

---

## 🧠 How It Works

```
┌─────────────┐
│  PDF upload │
└──────┬──────┘
       │
       ▼
┌──────────────────────┐   pdf_processor.py
│ 1. Extract text      │   PyPDF reads each page, keeping 1-based page numbers
└──────┬───────────────┘
       ▼
┌──────────────────────┐   text_chunker.py
│ 2. Split into chunks │   ~250-token pieces with ~50-token overlap
└──────┬───────────────┘
       ▼
┌──────────────────────┐   embedding_manager.py
│ 3. Create embeddings │   each chunk becomes 384 numbers (all-MiniLM-L6-v2)
└──────┬───────────────┘
       ▼
┌──────────────────────┐   vector_store.py
│ 4. Build FAISS index │   exact cosine-similarity search
└──────┬───────────────┘
       ▼
┌──────────────────────┐   qa_engine.py
│ 5. Retrieve + answer │   top 3 chunks → Gemini → cited answer
└──────────────────────┘
```

**Why the overlap matters:** chunk boundaries fall at arbitrary points. Without
overlap, a sentence split across two chunks could be lost from both. The
50-token overlap means any sentence survives whole in at least one chunk.

**Why 250 tokens:** an embedding is a single vector averaging everything in its
chunk. Larger chunks dilute specific details until search can no longer find
them. This project started at 500 tokens and measured a real improvement at
250 — the answer-bearing chunk moved from rank 15 to rank 3 on a test question.

---

## 🛠️ Technologies Used

| Technology | Version | Role |
|---|---|---|
| [Python](https://www.python.org/) | 3.13.2 | Language |
| [Streamlit](https://streamlit.io/) | 1.67.0 | Web interface |
| [pypdf](https://github.com/py-pdf/pypdf) | 7.0.0 | PDF text extraction |
| [Sentence Transformers](https://www.sbert.net/) | 6.2.0 | Local embeddings (`all-MiniLM-L6-v2`) |
| [FAISS](https://github.com/facebookresearch/faiss) | 1.15.1 | Vector similarity search |
| [google-genai](https://github.com/googleapis/python-genai) | 2.34.0 | Official Google Gemini SDK |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | 1.2.4 | Loads the API key from `.env` |
| [PyTorch](https://pytorch.org/) | 2.15.0 | Runs the embedding model (MPS on Apple Silicon) |

**Design choices worth noting:**

- **`IndexFlatIP`** (exact brute-force search) rather than an approximate index.
  For a document's worth of chunks it is instant and always exactly correct;
  approximate indexes exist for millions of vectors and would only add
  complexity here.
- **Normalised embeddings**, so inner product equals cosine similarity.
- **Local embedding model**, so indexing works offline and your document text
  is only sent to the API for the handful of retrieved passages.

---

## 📁 Project Structure

```
documind-ai/
├── app.py                      # Streamlit user interface
├── src/
│   ├── pdf_processor.py        # PDF → page text (with page numbers)
│   ├── text_chunker.py         # page text → overlapping chunks
│   ├── embedding_manager.py    # chunks → 384-dimensional vectors
│   ├── vector_store.py         # FAISS index + similarity search
│   └── qa_engine.py            # retrieval + Gemini → cited answer
├── sample_pdfs/
│   └── sample.pdf              # test document (public domain)
├── assets/
│   └── screenshots/            # screenshots for this README
├── .env                        # your API key — NEVER commit this
├── .env.example                # safe template, no secrets
├── .gitignore                  # excludes .env, venv/, caches
└── README.md
```

Each module does exactly one job and can be run on its own for testing:

```bash
python src/pdf_processor.py     sample_pdfs/sample.pdf   # extraction only
python src/text_chunker.py      sample_pdfs/sample.pdf   # + chunking
python src/embedding_manager.py sample_pdfs/sample.pdf   # + embeddings
python src/vector_store.py      sample_pdfs/sample.pdf   # + search
python src/qa_engine.py         sample_pdfs/sample.pdf "your question"
```

This layering makes the project easy to understand and debug — if something
breaks, you can find out exactly which stage is responsible.

---

## 💻 Installation (macOS)

### Prerequisites

- **macOS** (tested on Apple Silicon — M-series)
- **Python 3.11 or newer** (3.13 recommended)
- About **2 GB of free disk space** (PyTorch and the embedding model)

Check your Python version:

```bash
python3 --version
```

> **Note:** On macOS the command is `python3`, not `python`. If you have
> several versions installed, you can target one explicitly, e.g.
> `python3.13`. Avoid pre-release versions (anything ending in `rc`) — many
> packages do not publish builds for them yet.

### 1. Get the project

```bash
git clone https://github.com/shubhangisingh1158-design/documind-ai.git
cd documind-ai
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Your prompt should now begin with `(venv)`. That prefix is how you know the
environment is active — it keeps this project's packages separate from the rest
of your system.

> Re-activate with `source venv/bin/activate` every time you open a new
> Terminal window. Leave it with `deactivate`.

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install streamlit pypdf "google-genai<3.0.0" python-dotenv
python -m pip install sentence-transformers faiss-cpu
```

> The quotes around `"google-genai<3.0.0"` are **required** — without them the
> shell interprets `<` as a redirect and the command fails.

Installing in two steps is deliberate: the first four packages are small and
finish in seconds, while the last two pull in PyTorch (~2 GB) and take a few
minutes. Splitting them makes any problem easy to locate.

All packages ship native **arm64** wheels, so nothing compiles from source on
Apple Silicon.

### 4. Verify the installation

```bash
python -c "import streamlit, pypdf, dotenv, sentence_transformers, faiss; from google import genai; print('all imports OK')"
```

---

## 🔑 `.env` Configuration

The app reads your Gemini API key from a file called `.env` in the project
root. This keeps the key out of your source code.

### 1. Get a free API key

Visit **[Google AI Studio](https://aistudio.google.com/apikey)** and create an
API key. No payment details are required for the free tier.

### 2. Create your `.env` file

```bash
cp .env.example .env
```

### 3. Add your key

Open the file:

```bash
open -e .env
```

Set your key on a single line:

```env
GEMINI_API_KEY=your-api-key-here
```

**Formatting rules — all three matter:**

- ✅ No quotes around the value
- ✅ No spaces around the `=`
- ✅ Everything on one line, nothing after it

### Optional: choose a different model

Gemini model names change over time. The app uses a sensible default, but you
can override it:

```env
GEMINI_MODEL=models/gemini-3.5-flash
```

To see which models your key can access:

```bash
python -c "import sys; sys.path.insert(0,'src'); from qa_engine import list_available_models; print(list_available_models())"
```

If you ever see a "model not found" error, run that command and set one of the
names it prints.

> **Note:** `.env` files are hidden in Finder. Press **⌘ + Shift + .**
> (Command-Shift-period) to show hidden files. In Terminal, `ls -la` always
> shows them.

---

## 🚀 Running the Application

```bash
source venv/bin/activate
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**.

Press **Ctrl + C** in the Terminal to stop it.

> macOS may ask whether to allow incoming network connections. This is safe —
> Streamlit runs a local server and the traffic does not leave your machine.

### Using the app

1. **Upload a PDF** using the file picker. The sidebar shows the page count,
   how many pages contain text, and the number of chunks indexed.
2. **Ask a question** in the chat box at the bottom.
3. **Read the answer** — the source pages appear beneath it.
4. **Expand "Show the excerpts used"** to see the exact passages the answer was
   built from, with similarity scores.

### Try it with the sample document

The included `sample_pdfs/sample.pdf` is *Alice's Adventures in Wonderland*
(Chapters I–III, public domain) — 11 pages, 42 chunks.

| Question | What to expect |
|---|---|
| *What did Alice find on the little glass table?* | A tiny golden key (page 2) and later a bottle labelled "DRINK ME" (page 3) |
| *Who wrote this book?* | Lewis Carroll (page 1) |
| *Describe the Caucus-race.* | Cited answer from pages 9–10 |
| *What is the capital of Mongolia?* | *"The PDF does not contain enough information…"* — no citations |

That last one is the most important test: it shows the app refuses to answer
from outside the document.

### Troubleshooting

| Problem | Solution |
|---|---|
| `command not found: python` | Use `python3` on macOS |
| `zsh: no such file or directory: 3.0.0` | Add quotes: `"google-genai<3.0.0"` |
| "No API key found in .env" | Create `.env` and add `GEMINI_API_KEY=...` |
| "Gemini rejected your API key" | Get a fresh key from Google AI Studio; check for stray quotes or spaces |
| "The model ... was not found" | Run `list_available_models()` and set `GEMINI_MODEL` in `.env` |
| "No text could be extracted" | The PDF is scanned (images of pages) and needs OCR |
| First run is slow | The embedding model (~90 MB) downloads once, then is cached |

---

## 🔒 Security Note

**Never upload `.env` or `venv/` to GitHub.**

This project's `.gitignore` already excludes both, but it is worth
understanding *why* — this is the single most common and most costly mistake
beginners make with API keys.

### Why `.env` must stay private

Your `.env` file contains a **live API key tied to your Google account**.
If it reaches a public repository:

- Automated bots scan GitHub for leaked keys **within seconds** of a push.
- Anyone who finds it can make API calls billed to your account.
- **Deleting the file later does not help.** Git keeps every version in its
  history, so the key stays recoverable from previous commits forever.

The only real fix after a leak is to **revoke the key immediately** in Google
AI Studio and create a new one.

### Why `venv/` must stay out

It is roughly **1.5 GB** of installed packages, it is specific to your machine
and operating system, and it is fully rebuildable from the install commands
above. Committing it bloats the repository and helps nobody.

### The safe pattern this project uses

| File | Committed? | Contains |
|---|---|---|
| `.env` | ❌ **Never** | Your real API key |
| `.env.example` | ✅ Yes | An empty template showing which variables are needed |
| `venv/` | ❌ Never | Rebuildable packages |

Anyone cloning the repository copies `.env.example` to `.env` and adds their
own key. The template documents *what* is needed without revealing *your*
values.

### Before your first commit

Confirm the protection is working:

```bash
git status
```

**`.env` and `venv/` must not appear in the list of files to be committed.**
If either does, stop and check that `.gitignore` is present and committed
first.

You can also verify a specific file is ignored:

```bash
git check-ignore -v .env
```

If that prints a matching rule, the file is safely excluded.

### General good habits

- Never paste your API key into source code, a screenshot, an issue, or a chat
  message.
- Never print it to the terminal — this app only ever reports *whether* a key
  is configured, never its value.
- Treat a key that has ever been committed as compromised, and revoke it.

---

## 🗺️ Possible Improvements

Ideas for extending the project:

- **OCR support** for scanned PDFs (`pytesseract`)
- **Multi-document search** across several PDFs at once
- **Persist the index** to disk with `faiss.write_index()` so re-uploading is
  not needed
- **Highlight the source text** inside an embedded PDF viewer
- **Retrieval evaluation set** — a list of questions with known answers, to
  measure how often the correct page is retrieved

---

## 📄 License

Released under the MIT License — free to use, modify and learn from.

The sample PDF is *Alice's Adventures in Wonderland* by Lewis Carroll, sourced
from [Project Gutenberg](https://www.gutenberg.org/ebooks/11) and in the public
domain.

---

## Author

**Shubhangi Singh**

GitHub: https://github.com/shubhangisingh1158-design
