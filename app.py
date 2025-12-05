"""
app.py

Entry point for Hugging Face Spaces.

We reuse the UI definition from gradio_app.build_interface()
so we don't duplicate logic. Locally you can still run either:

    python gradio_app.py
or
    python app.py
"""

import gradio as gr
from gradio_app import build_interface


# Hugging Face looks for a top-level `demo` (or `app`) Gradio object.
demo = build_interface()


if __name__ == "__main__":
    # For local development / testing.
    demo.launch()
