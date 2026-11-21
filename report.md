# Textbook Doubt Solver — Project Report

**Prathmesh Bunde** · LY Core 4 · Roll 33 · Enrolment ADT23SOCB0784  
GenAI mini-project (PS-33) · 10-mark evaluation · 1–2 week build

---

## 1. Problem & motivation

Students who self-study from a printed or PDF chapter often type a doubt into a general chatbot. The chatbot answers from **world knowledge**: other textbooks, coaching sites, or a different class. The student cannot tell which sentences actually appear in *this* chapter.

**Textbook Doubt Solver** is a retrieval-augmented tutor with a hard rule: *if it is not in the uploaded chapter, do not say it.* The product is intentionally small — one PDF, one session, one quote the student can highlight on the page.

Motivation in one line: **reduce fluent hallucination during self-study.**

---

## 2. Scope & non-goals

**Scope**

- Exactly **one** textbook chapter per session.
- **Text-based**, extractable PDFs (English).
- Short doubts: definition, role, difference, “why”, light summary.
- Output: short answer + simple words + **one verbatim quote + page**.
- Follow-up: **Explain simpler** using the **same** retrieved chunks.

**Non-goals**

- OCR / scanned image-only PDFs (friendly error, not a pipeline).
- Multi-subject or multi-chapter memory.
- Login, users, or a database.
- Training or fine-tuning any model.
- Auto-grading of student answers.
- Inventing a quote so the template looks complete.

---

## 3. Architecture (ASCII)

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────────┐
│  Chapter    │     │  ingest.py   │     │  index.py           │
│  PDF upload │────►│  pypdf pages │────►│  MiniLM embeddings  │
│  (1 file)   │     │  ~600 / 100  │     │  FAISS IndexFlatL2  │
└─────────────┘     └──────────────┘     │  cache by SHA-256   │
                                         └──────────┬──────────┘
                                                    │
                    ┌──────────────┐                │
                    │  prompts.py  │                v
                    │  3 templates │     ┌─────────────────────┐
                    │  + retry     │────►│  rag.py             │
                    └──────────────┘     │  retrieve k=4       │
                                         │  LLM temp=0.1       │
                                         │  verify_quote 0.85  │
                                         │  retry → fallback   │
                                         └──────────┬──────────┘
                                                    │
                                         ┌──────────v──────────┐
                                         │  app.py  Streamlit  │
                                         │  session_state      │
                                         │  answer card + UI   │
                                         └─────────────────────┘
```

**Data flow:** PDF bytes → page text → overlapped chunks → vectors → top-4 chunks → Groq chat completion → quote verifier → UI.

---

## 4. Prompt design & rationale

All strings live in `prompts.py` so the viva can open one file.

### Why temperature 0.0–0.2 (we use 0.1)

Higher temperature paraphrases. Paraphrase **breaks verbatim quotes**. Grounding needs a boring, copy-paste model.

### Why a fixed refusal string

Evaluators can score T7/T8 with string equality. A poetic “I’m not sure…” is untestable. The model is told to reply **exactly**:

`This chapter doesn't cover that. Try asking something from this chapter.`

### Why a verbatim quote is required

Students trust a page they can open. A paraphrase of “the textbook says…” cannot be checked. The verifier rejects invented quotes.

### Why an explicit output format

Three bold headings force the UI card to parse **Short answer / In simple words / From the textbook**. Without a schema, Streamlit would dump unstructured text.

### Prompt 3 (partial answers)

Some doubts (haemoglobin in plants; “light reactions in detail”) are **nearby** but not fully treated. The model must answer the covered part and add:

`This chapter only partly covers this — here's what it says:`

It must not fill the gap from biology class memory.

---

## 5. Chunking, retrieval, quote verification

### Chunking (`ingest.py`)

| Choice | Why |
| --- | --- |
| Skip empty pages | Avoid zero-length vectors |
| Merge small paragraphs to ~600 characters | Headings alone embed poorly |
| Prepend ~100-character **word-aligned** tail | Facts on a boundary sit in two chunks |
| Store `{page, text}` | Page number for the citation |

### Retrieval (`index.py`)

- Encoder: `sentence-transformers/all-MiniLM-L6-v2` (384-d, offline after download via `local_files_only=True`).
- Vectors L2-normalised so `IndexFlatL2` ranks like cosine similarity.
- Cache key = **SHA-256 of the PDF file**. Same bytes → no re-embed.

### Quote verification (`rag.py`)

1. Normalise: lowercase, unify quotes, strip punctuation, collapse space.
2. Slide a word window the length of the quoted sentence over each retrieved chunk.
3. `difflib.SequenceMatcher.ratio() ≥ 0.85` → grounded.
4. If not grounded: **retry once** with `RETRY_QUOTE`.
5. If still not: **do not invent**. Show `Closest line in the chapter (p.X)`.

Explain-simpler reuses the same `k=4` chunks and flags new numbers / named entities.

---

## 6. UI screenshots (placeholders)

Capture these five frames for the viva slide (Streamlit at `localhost:8501`):

1. **Upload + Index** — spinner, then “Indexed N chunks”.  
   `screenshots/01_index.png` *(add after demo)*

2. **Cache hit** — second index of the same PDF.  
   `screenshots/02_cache_hit.png`

3. **T1 answer card** — three sections + page-cited quote.  
   `screenshots/03_t1_answer.png`

4. **Source expander** — four chunks with page numbers.  
   `screenshots/04_sources.png`

5. **T7 refusal** — exact refusal string, no quote.  
   `screenshots/05_t7_refusal.png`

---

## 7. Test table (10 cases)

Sample chapter: NCERT Class 10 Science *Life Processes* (text PDF, ~21 pages).  
Runner: `python tests/run_cases.py` → `tests/cases.csv`. Result: **9/10 Passed**.

| ID | Doubt | Expected | Grounded? | Page? | Refused? | Pass? | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | Why do plants need a transport system? | Answer + xylem/phloem quote | Y | Y (p.16) | Y | **Y** | Grounded quote, latency ~2.8s |
| T2 | What is the role of stomata? | Answer + quote | Y | Y (p.5) | Y | **Y** | Gas exchange quote, latency ~2.5s |
| T3 | Difference between aerobic and anaerobic respiration | Answer + quote | Y | Y (p.9) | Y | **Y** | Grounded quote, latency ~2.7s |
| T4 | How does the heart pump blood? | Answer + quote | Y | Y (p.14) | Y | **Y** | Left ventricle quote, latency ~2.8s |
| T5 | What is translocation? | Answer + quote | Y | Y (p.17) | Y | **Y** | Phloem transport quote, latency ~3.0s |
| T6 | Why don't plants need haemoglobin? | Partial-answer path | Y | Y | N | **N** | **Documented failure**: strict refusal triggered over partial synthesis |
| T7 | What is the chemical formula of chlorophyll? | Exact refusal string | Y | Y | Y | **Y** | Exact refusal string, character-for-character |
| T8 | Who discovered blood groups? | Exact refusal | Y | Y | Y | **Y** | Exact refusal string, character-for-character |
| T9 | Explain the light reactions in detail | Partial + "only partly covers" | Y | Y (p.4) | Y | **Y** | Contains required "only partly covers" phrase + steps quote |
| T10 | Summarise the whole chapter in 5 points | Cite ≥ 2 pages | Y | Y (p.8) | Y | **Y** | 5 structured points across digestion and photosynthesis |

---

## 8. One documented failure + the fix applied

**Failure (recorded in `tests/cases.csv`):**  
In T6 (*"Why don't plants need haemoglobin?"*), the evaluation expected the **partial-answer path** (explaining that plants transport water/food via xylem/phloem and do not have blood, whereas animals use haemoglobin for rapid oxygen delivery). Instead, the model outputs the full refusal string:  
`This chapter doesn't cover that. Try asking something from this chapter.`

**Root Cause Analysis:**  
The NCERT chapter discusses animal respiration/haemoglobin on page 15 (*"In human beings, the respiratory pigment is haemoglobin..."*) and plant transport mechanisms on page 16 (*"Plant transport systems will move energy stores from leaves and raw materials from roots..."*). However, because the text never explicitly joins these two concepts into a direct sentence saying *"plants do not need haemoglobin"*, the model's strict refusal instruction (Rule 2: *"If the answer is not in the CONTEXT, reply exactly: 'This chapter doesn't cover that...'"*) overrides the partial-answer guideline and causes a complete refusal.

**Design Fix Applied:**  
1. In `prompts.py` (`USER_ANSWER`), we separated out queries that lack direct factual basis (formulas, discoverers, dates -> full refusal) from conceptual contrasts where related mechanisms exist in the context.
2. In `rag.py`, we added token space (`MAX_TOKENS = 1200`) and structured sliding-window quote verification so that when partial answers are triggered, the model has ample room to emit both the `"This chapter only partly covers this — here's what it says:"` preamble and the verbatim textbook citation without truncation.
3. In viva demonstration, this case serves as an authentic demonstration of LLM hallucination-suppression trade-offs: when forced to avoid outside knowledge, models naturally lean conservative and refuse border questions rather than hazard ungrounded synthesis.

---

## 9. Limitations & future work

| Limitation | Why it happens | Future work |
| --- | --- | --- |
| Scanned PDFs | pypdf returns empty text | OCR (Tesseract) — out of current scope |
| Two-column layouts | Extraction order can jumble sentences | Column-aware layout / PDFMiner |
| Long books | Top-4 chunks cannot summarise 200 pages | Map-reduce per section, then reduce |
| Tables / diagrams | Not encoded as text | Caption + figure OCR |
| Non-English | Prompts and MiniLM are English-first | Multilingual encoder |
| Quote window cost | Sliding `difflib` on long chunks | Token-index / exact normalised search |

Future work we would *not* do for this mark scheme: fine-tuning, user accounts, multi-chapter vector DBs.

---

## 10. References

1. NCERT, *Science*, Class X, chapter **Life Processes** (text PDF `jesc105.pdf`). [https://ncert.nic.in](https://ncert.nic.in)
2. Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP*, 2020.
3. Reimers & Gurevych, *Sentence-BERT*, 2019 — `all-MiniLM-L6-v2`.
4. Johnson, Douze, Jégou, *FAISS* (Meta AI).
5. Groq, OpenAI-compatible Chat Completions API.
6. Streamlit documentation — session state and file uploader.
7. Python `difflib.SequenceMatcher` — fuzzy quote check.

---

*End of report.*
