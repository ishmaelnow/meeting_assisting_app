"""
qa.py

Simple RAG-style Question Answering over a meeting transcript.

We assume:
- You already built a retriever from the transcript (FAISS + OpenAI embeddings).
- You pass that retriever + a question into the function below.

The function:
- Uses the retriever to get relevant chunks.
- Sends those chunks + question to Mistral.
- Returns a grounded, concise answer.
"""

from typing import Any, List

from langchain_core.documents import Document


from config import mistral_client, MISTRAL_CHAT_MODEL


def _ensure_mistral_client():
    """
    Make sure the Mistral client is configured.
    """
    if mistral_client is None:
        raise RuntimeError(
            "Mistral client is not configured. "
            "Make sure MISTRAL_API_KEY is set in your environment/.env file."
        )


def _format_context(docs: List[Document], max_chars: int = 4000) -> str:
    """
    Turn a list of Documents into a single context string.

    We also truncate to max_chars so we don't exceed prompt limits.
    """
    parts: List[str] = []
    current_len = 0

    for doc in docs:
        text = doc.page_content.strip()
        if not text:
            continue

        # Add a separator so Mistral can see boundaries
        chunk = f"- {text}\n"
        if current_len + len(chunk) > max_chars:
            break

        parts.append(chunk)
        current_len += len(chunk)

    if not parts:
        return "No relevant context was found in the transcript."

    return "".join(parts)


def answer_question_with_mistral(
    question: str,
    retriever: Any,
    max_context_chars: int = 4000,
) -> str:
    """
    Answer a question about the meeting using:

    - retriever (FAISS) to fetch relevant transcript chunks
    - Mistral LLM to generate a grounded answer

    Parameters
    ----------
    question : str
        The user's question about the meeting.
    retriever : Any
        A LangChain retriever created from your FAISS vector store.
    max_context_chars : int
        Maximum number of characters of context to pass to Mistral.

    Returns
    -------
    str
        Mistral's answer.
    """
    _ensure_mistral_client()

    # 1) Use retriever to get relevant transcript chunks
    docs: List[Document] = retriever.invoke(question)


    if not docs:
        return "I couldn't find anything in the transcript related to that question."

    context_text = _format_context(docs, max_chars=max_context_chars)
#                                   ^^^^^^^^


    # 2) Build prompt for Mistral
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful Meeting Assistant. "
                "You answer questions ONLY using the provided meeting context. "
                "If the answer is not in the context, say you don't know."
            ),
        },
        {
            "role": "user",
            "content": (
                "Here is context from the meeting transcript:\n\n"
                f"{context_text}\n\n"
                "Using ONLY this context, answer the following question:\n\n"
                f"Question: {question}"
            ),
        },
    ]

    # 3) Call Mistral
    try:
        response = mistral_client.chat.complete(
            model=MISTRAL_CHAT_MODEL,
            messages=messages,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to get answer from Mistral: {e}") from e

    try:
        answer_text = response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Unexpected response format from Mistral: {e}") from e

    return answer_text
