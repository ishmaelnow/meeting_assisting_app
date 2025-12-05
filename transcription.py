"""
transcription.py

Functions to transcribe audio using OpenAI Whisper via the OpenAI Python client.

We reuse:
- openai_client
- OPENAI_WHISPER_MODEL

from config.py
"""

from typing import BinaryIO

from config import openai_client, OPENAI_WHISPER_MODEL


def transcribe_audio_path(file_path: str) -> str:
    """
    Transcribe an audio file from disk using Whisper.

    Parameters
    ----------
    file_path : str
        Path to the audio file, e.g. "meeting.wav" or "recording.mp3"

    Returns
    -------
    str
        The transcribed text.

    Raises
    ------
    RuntimeError
        If the API call fails.
    """
    try:
        # Open the file in binary mode ("rb" = read binary)
        with open(file_path, "rb") as audio_file:
            response = openai_client.audio.transcriptions.create(
                model=OPENAI_WHISPER_MODEL,
                file=audio_file,
            )
        # The OpenAI client returns an object with a `.text` attribute
        return response.text
    except Exception as e:
        # Wrap lower-level exceptions in a clear message
        raise RuntimeError(f"Failed to transcribe audio from path '{file_path}': {e}") from e


def transcribe_audio_filelike(audio_file: BinaryIO) -> str:
    """
    Transcribe an audio file-like object (for example from Gradio).

    This is useful when Gradio passes you something like:
      - a temporary file object, or
      - an uploaded file handle

    Parameters
    ----------
    audio_file : BinaryIO
        A file-like object opened in binary mode.

    Returns
    -------
    str
        The transcribed text.

    Raises
    ------
    RuntimeError
        If the API call fails.
    """
    try:
        response = openai_client.audio.transcriptions.create(
            model=OPENAI_WHISPER_MODEL,
            file=audio_file,
        )
        return response.text
    except Exception as e:
        raise RuntimeError("Failed to transcribe audio from file-like object") from e
