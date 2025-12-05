"""
config.py

Central place to configure:
- API keys
- Model names
- Shared clients for OpenAI and Mistral

We load everything from environment variables so we NEVER hard-code keys.
"""

import os
from dotenv import load_dotenv

# 🧪 1. Load variables from .env file (if present)
#    Example .env (same folder as this file):
#      OPENAI_API_KEY=sk-...
#      OPENAI_WHISPER_MODEL=whisper-1
#      OPENAI_EMBEDDING_MODEL=text-embedding-3-small
#      MISTRAL_API_KEY=...
#      MISTRAL_CHAT_MODEL=mistral-small-latest
load_dotenv()


# 🟦 2. OpenAI settings
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENAI_WHISPER_MODEL: str = os.getenv("OPENAI_WHISPER_MODEL", "whisper-1")
OPENAI_EMBEDDING_MODEL: str = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",  # good default, cheap & strong
)

# 🟩 3. Mistral settings
MISTRAL_API_KEY: str | None = os.getenv("MISTRAL_API_KEY")
MISTRAL_CHAT_MODEL: str = os.getenv(
    "MISTRAL_CHAT_MODEL",
    "mistral-small-latest",  # you can change this later
)


# 🔒 4. Basic safety checks (optional but helpful for debugging)
if OPENAI_API_KEY is None:
    raise ValueError(
        "OPENAI_API_KEY is not set. "
        "Create a .env file or set the environment variable before running the app."
    )

if MISTRAL_API_KEY is None:
    # We don't raise here, because you might want to test only Whisper/embeddings first.
    # But we warn loudly.
    print(
        "[config] WARNING: MISTRAL_API_KEY is not set. "
        "Mistral-based summarization/Q&A will fail until you set it."
    )


# 🧠 5. Create shared API clients
#    - OpenAI client (for Whisper + embeddings + optionally chat)
#    - Mistral client (for chat/summarization/Q&A)

from openai import OpenAI  # official OpenAI Python client
from mistralai import Mistral  # official Mistral Python client


# Single OpenAI client used across your app
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Single Mistral client used across your app
mistral_client = Mistral(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None
