"""
summarization.py

Summarize a meeting transcript using Mistral.

We support:
- Short transcripts: summarize directly in one call.
- Long transcripts: summarize in chunks, then summarize the summaries
  (a simple "map-reduce" style).

Relies on:
- mistral_client
- MISTRAL_CHAT_MODEL

from config.py
"""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter


from config import mistral_client, MISTRAL_CHAT_MODEL


def _ensure_mistral_client():
    """
    Internal helper: make sure the Mistral client is available.

    Raises
    ------
    RuntimeError
        If the Mistral client is not configured.
    """
    if mistral_client is None:
        raise RuntimeError(
            "Mistral client is not configured. "
            "Make sure MISTRAL_API_KEY is set in your environment/.env file."
        )


def _call_mistral_summary(prompt_content: str) -> str:
    """
    Low-level helper that sends a summary request to Mistral.

    Parameters
    ----------
    prompt_content : str
        The text we want Mistral to summarize (transcript or partial transcript).

    Returns
    -------
    str
        The summary text returned by Mistral.

    Raises
    ------
    RuntimeError
        If the API call fails or returns an unexpected response.
    """
    _ensure_mistral_client()

    # We build a simple chat-style prompt:
    # - system: "You are a meeting assistant"
    # - user: "Here is the transcript, please summarize it in a structured way"
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful Meeting Assistant. "
                "Your job is to read meeting transcripts and produce concise, "
                "structured summaries that highlight key points, decisions, "
                "and action items."
            ),
        },
        {
            "role": "user",
            "content": (
                "Here is a meeting transcript (or part of it). "
                "Please summarize it in the following structure:\n\n"
                "1. Overview (2–3 sentences)\n"
                "2. Key decisions (bullet points)\n"
                "3. Action items (bullet points, include owner and due date if mentioned)\n\n"
                "Transcript:\n"
                f"{prompt_content}"
            ),
        },
    ]

    try:
        response = mistral_client.chat.complete(
            model=MISTRAL_CHAT_MODEL,
            messages=messages,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to get summary from Mistral: {e}") from e

    # The Mistral Python client returns a response object with choices
    # Each choice has a "message" with "content"
    try:
        summary_text = response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"Unexpected response format from Mistral: {e}") from e

    return summary_text


def _split_text(text: str, chunk_size: int = 4000, chunk_overlap: int = 400) -> List[str]:
    """
    Split a long transcript into smaller chunks of text.

    Uses LangChain's RecursiveCharacterTextSplitter to try to break
    on nice boundaries (paragraphs, sentences, etc.).

    Parameters
    ----------
    text : str
        The full transcript text.
    chunk_size : int
        Approximate character size of each chunk.
    chunk_overlap : int
        Overlap in characters between chunks to preserve context.

    Returns
    -------
    List[str]
        A list of string chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],  # try larger breaks first
    )

    chunks = splitter.split_text(text)
    return chunks


def summarize_meeting_with_mistral(transcript: str, max_direct_chars: int = 5000) -> str:
    """
    Summarize a meeting transcript using Mistral.

    Behavior:
    - If the transcript is short enough (<= max_direct_chars), send it in one go.
    - If it is longer, we:
        1) Split into chunks
        2) Summarize each chunk
        3) Summarize the summaries into a final high-level summary

    Parameters
    ----------
    transcript : str
        Full meeting transcript text.
    max_direct_chars : int
        Character length threshold for "short" transcripts.
        Above this, we use the chunked (map-reduce style) approach.

    Returns
    -------
    str
        A structured summary of the meeting.
    """
    # Strip leading/trailing whitespace to avoid useless tokens
    transcript = transcript.strip()

    if not transcript:
        return "No transcript content provided to summarize."

    # Case 1: short transcript → single Mistral call
    if len(transcript) <= max_direct_chars:
        return _call_mistral_summary(transcript)

    # Case 2: long transcript → chunked summarization

    # 1) Split into chunks
    chunks = _split_text(transcript, chunk_size=max_direct_chars, chunk_overlap=300)

    # 2) Summarize each chunk separately
    partial_summaries: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        print(f"[summarization] Summarizing chunk {idx}/{len(chunks)}...")
        partial_summary = _call_mistral_summary(chunk)
        partial_summaries.append(
            f"Chunk {idx} summary:\n{partial_summary}"
        )

    # 3) Combine chunk summaries into one big text
    combined_summaries_text = "\n\n".join(partial_summaries)

    # 4) Ask Mistral to summarize the summaries into a final overview
    final_messages_text = (
        "You have been given summaries of different parts of a long meeting. "
        "Please combine them into ONE overall meeting summary with the "
        "same structured format:\n\n"
        "1. Overview (2–3 sentences)\n"
        "2. Key decisions (bullet points)\n"
        "3. Action items (bullet points)\n\n"
        "Here are the partial summaries:\n\n"
        f"{combined_summaries_text}"
    )

    final_summary = _call_mistral_summary(final_messages_text)
    return final_summary
