"""The answering step shared by every system: the same model and prompt answer from the recalled context."""

from __future__ import annotations

ANSWER_SYSTEM = 'You are a helpful assistant with long-term memory of past conversations with the user.'
ANSWER_PROMPT = "Below is what your memory system retrieved about past conversations with the user.\n\nMEMORY:\n{context}\n\nCurrent date: {date}\nQuestion: {question}\n\nAnswer the question using the memory. Dates in brackets are when each conversation happened. If the memory does not contain enough information, say that you don't know. Answer concisely."
ANSWER_MAX_TOKENS = 300


def answer(llm, question: str, date: str, context: str) -> str:
    prompt = ANSWER_PROMPT.format(context=context or "(nothing retrieved)", date=date, question=question)
    return llm.complete(prompt, system=ANSWER_SYSTEM, max_tokens=ANSWER_MAX_TOKENS).text
