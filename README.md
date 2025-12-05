# Meeting Assisting App

Transcribe meeting audio with Whisper, summarize with Mistral, and do RAG-style Q&A over the transcript using FAISS + LangChain.

## Quick Start

```bash
# 1. Create & activate venv (example)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Put a small audio file in the project folder, e.g.:
#    min_meeting.mp3

# 4. Set environment variables (or .env)
# OPENAI_API_KEY=...
# MISTRAL_API_KEY=...

# 5. Run
python main.py
