# Textbook Doubt Solver

**Grounded RAG tutor for a single textbook chapter.**

A student uploads **one** chapter PDF, indexes it, and asks doubts. The app answers **only** from that chapter: a short explanation, an everyday rephrase, and a **verbatim quote with a page number**. If the chapter does not contain the answer, it refuses with a fixed sentence. Ungrounded quotes never reach the UI.

| Field | Detail |
| --- | --- |
| Student | **Prathmesh Bunde** |
| Class | LY · Core 4 · Roll **33** |
| Problem | GenAI mini-project — Textbook Doubt Solver (PS-33) |
| Enrolment | ADT23SOCB0784 |
| Stack | Python · Streamlit · pypdf · MiniLM · FAISS · Groq LLM · difflib |
| Sample chapter | NCERT Class 10 Science — *Life Processes* (current book Ch. 5 / older books Ch. 6) |

---

## Why this exists

Self-study fails when a chatbot **sounds confident and is wrong**. Generic LLMs mix other grades, other boards, and memorised internet facts into the answer. This project treats the uploaded chapter as the **only** allowed knowledge source.

Design goals for a 10-mark viva:

1. **Grounding** — every claim is retrieved from the PDF, not from model memory.
2. **Citations students can check** — one verbatim line plus page number.
3. **Honest refusal** — out-of-chapter questions get one fixed sentence, not a guess.
4. **No training** — embeddings are off-the-shelf MiniLM; the LLM is an API.

---

## What you get in the UI

- Upload **one** chapter PDF (or use the bundled NCERT sample).
- **Index chapter** with a spinner (SHA-256 cache makes re-uploads instant).
- Question box; **Ask** stays disabled on empty / whitespace input.
- Answer card with three blocks: **Short answer**, **In simple words**, **From the textbook**.
- **Explain simpler** — same retrieved chunks, no new facts.
- Source expander: retrieved chunks, page numbers, FAISS L2 scores.
- Latency (via `time.perf_counter()`), quote-grounded flag, retry/fallback path.
- **Clear / reset** for a new chapter (one chapter per session).

Friendly errors:

| Situation | What the student sees |
| --- | --- |
| Scanned / image-only PDF | `This PDF has no extractable text; OCR is out of scope` |
| Empty pages | Skipped silently during ingest |
| Quote not in the chapter | Retry once, then `Closest line in the chapter (p.X)` |
| Doubt not in the chapter | Exact refusal string (see below) |

**Exact refusal string** (character-for-character):

```text
This chapter doesn't cover that. Try asking something from this chapter.
```

---

## Architecture

```text
 PDF (1 chapter)
        |
        v
  ingest.py                 pypdf  ->  page-tagged paragraphs
  load_chunks()             merge small paras (~600 chars)
                            prepend 100-char tail overlap
        |
        v
  index.py                  all-MiniLM-L6-v2 (offline after first download)
  SHA-256(pdf) cache        FAISS IndexFlatL2
        |
        v
  rag.py  retrieve(k=4)  -->  Groq LLM (temperature 0.1)
              |                    |
              |                    v
              |              prompts.py  (all templates)
              |                    |
              +-- verify_quote() difflib, threshold 0.85
                      |  fail
                      v
                 retry once --> still fail --> labelled closest line
        |
        v
  app.py  Streamlit  (session_state: chunks, index, last answer, last context)
```

---

## Repository layout

```text
llm-ps33-ADT23SOCB0784/
├── app.py                 # Streamlit UI
├── ingest.py              # PDF -> page-tagged chunks
├── index.py               # MiniLM embeddings + FAISS + hash cache
├── rag.py                 # retrieve, answer, verify, retry, simplify
├── prompts.py             # ALL prompt strings (nothing inline in rag.py)
├── requirements.txt
├── .env.example           # copy to .env and add GROQ_API_KEY
├── .gitignore
├── README.md
├── report.md              # 10-section evaluation report
├── data/
│   └── ncert_class10_science_ch6_life_processes.pdf
└── tests/
    ├── cases.csv          # 10 viva cases with pass/fail
    └── run_cases.py       # batch runner
```

Runtime folders (gitignored): `cache/` (FAISS + upload copies), `.venv/`, `.env`.

---

## Install

Python **3.10+** (tested on Windows 10 / Python 3.13).

```bash
git clone https://github.com/Sparkydev007/llm-ps33-ADT23SOCB0784.git
cd llm-ps33-ADT23SOCB0784

python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Packages: `streamlit`, `pypdf`, `sentence-transformers`, `faiss-cpu`, `openai`, `python-dotenv`, `numpy`.

`difflib` is in the Python standard library — **do not pip-install it**.

The first MiniLM load downloads `all-MiniLM-L6-v2` (~80 MB). After that it runs **offline**. The LLM still needs the Groq API.

---

## API key (Groq)

1. Create a key at [https://console.groq.com](https://console.groq.com).
2. Copy `.env.example` to `.env`:

```bash
copy .env.example .env     # Windows
# cp .env.example .env     # macOS / Linux
```

3. Put the key in `.env` (never commit this file):

```env
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

The OpenAI Python SDK talks to Groq’s OpenAI-compatible endpoint:

`https://api.groq.com/openai/v1`

Temperature is fixed at **0.1** (allowed range 0–0.2) so quotes stay stable.

---

## Run the app

```bash
streamlit run app.py
```

Browser: `http://localhost:8501`

### 3-minute demo script (viva)

| Step | Action | What to point at |
| --- | --- | --- |
| 1 | Enable **Use sample** (or upload the NCERT PDF) | One chapter only |
| 2 | Click **Index chapter** | Spinner; chunk count; SHA-256 |
| 3 | Click **Index chapter** again | **Cache hit** — re-upload is instant |
| 4 | Ask **T1**: *Why do plants need a transport system?* | Quote names xylem/phloem + page |
| 5 | Open **Source chunks** | Retrieved text + page numbers |
| 6 | Click **Explain simpler** | Same facts; everyday analogy; same quote |
| 7 | Ask **T7**: *What is the chemical formula of chlorophyll?* | Exact refusal string |
| 8 | Open `tests/cases.csv` | 10 rows; documented failure + fix in `report.md` §8 |

---

## Run tests

Smoke ingest (chunk count + page histogram):

```bash
python ingest.py data/ncert_class10_science_ch6_life_processes.pdf
```

Smoke RAG (T1 quote verifier + T7 refusal):

```bash
python rag.py data/ncert_class10_science_ch6_life_processes.pdf
```

Full 10-case evaluation (writes `tests/cases.csv`):

```bash
python tests/run_cases.py
```

Acceptance checks:

| IDs | Must happen |
| --- | --- |
| T1–T5, T10 | Answer contains a quote that `verify_quote` accepts (≥ 0.85) |
| T7, T8 | Response **equals** the refusal string |
| T6, T9 | Partial path + phrase `only partly covers` |
| Explain simpler | No new numbers / named entities vs the first answer |
| Latency | ≤ 6 s per doubt after indexing on ~20 pages |

---

## How grounding is enforced

1. **Chunking** (`ingest.py`) — pages extracted with pypdf; empty pages dropped; small paragraphs merged to ~600 characters; last ~100 characters of the previous block are prepended so facts on a boundary appear in **two** chunks.
2. **Retrieval** (`index.py` + `rag.py`) — MiniLM embeddings, L2-normalised, FAISS `IndexFlatL2`, top **k = 4**.
3. **Prompts** (`prompts.py`) — system tutor prompt + partial-answer rule + explain-simpler template. `rag.py` has **no** inline prompt strings.
4. **Quote check** — extract `"…" (p.N)`, normalise, slide a word window over retrieved chunks, `difflib.SequenceMatcher` ≥ **0.85**.
5. **Ungrounded path** — retry once with `RETRY_QUOTE`. If it still fails, replace the fake quote with:

   `Closest line in the chapter (p.X): "…"`

6. **Cache** — `SHA-256(pdf bytes)` → `cache/<hash>/index.faiss` + `chunks.json`.

---

## Scope and non-goals

**In scope:** one English, text-based chapter; extractable PDF text; short doubts; page-cited quotes.

**Out of scope:** OCR, multi-chapter memory, login, database, model training/fine-tuning, auto-grading, inventing a quote to fill the template.

---

## Report

The marked write-up is [`report.md`](report.md): problem, scope, ASCII architecture, prompt rationale, chunking/verification, screenshot placeholders, 10-case table, one documented failure + fix, limitations, references.

---

## Security

- `.env` is gitignored. Rotate any key that was ever pasted in chat or a screenshot.
- Do not commit `GROQ_API_KEY`.
- The app never sends the whole PDF — only the top-4 retrieved chunks.

---

## License / academic use

Course submission for GenAI PS-33. Sample PDF is an NCERT textbook chapter; copyright remains with NCERT. Use for education and evaluation only.

**Repository:** [Sparkydev007/llm-ps33-ADT23SOCB0784](https://github.com/Sparkydev007/llm-ps33-ADT23SOCB0784)
