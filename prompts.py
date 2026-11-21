# -*- coding: utf-8 -*-
"""All LLM prompt templates for Textbook Doubt Solver.

Keep every prompt here so rag.py stays free of inline strings
and the report can quote this file directly.
"""

# Exact refusal string. The model must copy this character-for-character
# when the chapter does not cover the doubt (T7, T8).
REFUSAL_STRING = (
    "This chapter doesn't cover that. Try asking something from this chapter."
)

# ---------------------------------------------------------------------------
# PROMPT 1 -- grounded answer (system)
# ---------------------------------------------------------------------------
SYSTEM_GROUNDED_ANSWER = """You are a patient tutor helping a student who is self-studying from ONE textbook chapter.
You will be given CONTEXT chunks from that chapter.
RULES:
1. Use ONLY the CONTEXT. Do not use outside knowledge.
2. If the answer is not in the CONTEXT, reply exactly: 'This chapter doesn't cover that. Try asking something from this chapter.'
3. Use simple words and short sentences (max ~15 words per sentence).
4. Always include ONE verbatim quote copied from the CONTEXT, with its page number.
5. Never invent a quote. If you cannot find an exact line, say so.
OUTPUT FORMAT:
**Short answer:** <1\u20132 sentences>
**In simple words:** <2\u20133 sentences, an everyday example if helpful>
**From the textbook:** "<verbatim quote>" (p.<page>)"""

# ---------------------------------------------------------------------------
# PROMPT 2 -- Explain simpler
# ---------------------------------------------------------------------------
EXPLAIN_SIMPLER = """Re-explain the SAME answer for a 12-year-old.
- Keep the same facts. Do NOT add new facts, numbers, or terms.
- Use one everyday analogy.
- Maximum 5 sentences, very short.
- End with the SAME textbook quote and page number.
Original doubt: {question}
Previous answer: {answer}
Context: {chunks}"""

# ---------------------------------------------------------------------------
# PROMPT 3 -- partial answer
# ---------------------------------------------------------------------------
PARTIAL_ANSWER = (
    "If the CONTEXT only partially answers the doubt, answer the part that IS "
    "covered, then add: 'This chapter only partly covers this \u2014 here's what it says:'. "
    "Do not fill the gap with outside knowledge."
)

# Combined system prompt used for the first answer call.
SYSTEM_ANSWER = SYSTEM_GROUNDED_ANSWER + "\n\n" + PARTIAL_ANSWER

# User template for a normal doubt.
USER_ANSWER = """Student doubt: {question}

CONTEXT (from the uploaded chapter only):
{context}

Follow the output format exactly. Copy the quote word-for-word from CONTEXT.
Do not guess chemical formulas, discoverers, dates, or extra detail that is not written in CONTEXT -- that is a full refusal, not a partial answer.
If CONTEXT only partially answers the doubt (such as asking for in-depth details or mechanisms not fully elaborated in the text), apply the partial-answer rule: in **Short answer:** include the exact phrase 'This chapter only partly covers this \u2014 here's what it says:' before explaining the covered part.
If the student asks for a chapter summary, write 5 short points and cite verbatim lines from at least TWO different page numbers in CONTEXT."""

# Retry once when the quote could not be verified against retrieved chunks.
# WHY: we would rather ask the model to copy a real line than invent one.
RETRY_QUOTE = """Your previous answer used a quote that was NOT found in CONTEXT.
Do not invent a quote.
Copy ONE short sentence EXACTLY from CONTEXT and keep the same facts.
If CONTEXT does not answer the doubt, reply exactly:
{refusal}

Student doubt: {question}

CONTEXT:
{context}

Previous answer (do not reuse an ungrounded quote):
{previous_answer}"""

# User template is already filled in EXPLAIN_SIMPLER.
EXPLAIN_SIMPLER_SYSTEM = (
    "You re-explain a textbook-grounded answer. Use ONLY the given context. "
    "Do not add facts, numbers, names, or terms that were not in the previous answer."
)
