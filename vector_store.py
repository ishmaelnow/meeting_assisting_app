"""
vector_store.py

Builds a vector store and retriever from a meeting transcript using:

- LangChain text splitter
- OpenAI embeddings
- FAISS vector store

This will be the "memory" that our Meeting Assistant searches.
"""

from typing import Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

from config import OPENAI_EMBEDDING_MODEL


def _split_transcript_to_documents(
    transcript: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Split the raw transcript text into smaller chunks.

    Why?
    ----
    - Long transcripts don't fit nicely into a single LLM prompt.
    - Smaller chunks make retrieval more accurate.

    Parameters
    ----------
    transcript : str
        Full meeting transcript text.
    chunk_size : int
        Target size (characters) of each chunk.
    chunk_overlap : int
        How many characters to overlap between chunks to keep context.

    Returns
    -------
    list[Document]
        A list of LangChain Document objects representing transcript chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],  # try to split on sensible boundaries
    )

    # We wrap the whole transcript in a single Document first
    base_doc = Document(page_content=transcript, metadata={"source": "meeting_transcript"})

    docs = splitter.split_documents([base_doc])
    return docs


def build_vector_store_from_transcript(
    transcript: str,
) -> Tuple[FAISS, OpenAIEmbeddings]:
    """
    Build a FAISS vector store from a meeting transcript.

    Steps:
    1. Split transcript into Document chunks.
    2. Create OpenAI embeddings.
    3. Use FAISS to store vectors + documents.

    Parameters
    ----------
    transcript : str
        Full meeting transcript text.

    Returns
    -------
    (vector_store, embeddings)
        vector_store : FAISS
            The in-memory FAISS vector store with all chunks indexed.
        embeddings : OpenAIEmbeddings
            The embeddings object (useful if you want to reuse it later).
    """
    # 1) Split into chunks
    docs = _split_transcript_to_documents(transcript)

    # 2) Configure embeddings (this will call the OpenAI embeddings API under the hood)
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

    # 3) Build FAISS vector store from documents
    vector_store = FAISS.from_documents(docs, embeddings)

    return vector_store, embeddings


def build_retriever(vector_store: FAISS, k: int = 4):
    """
    Create a retriever from an existing vector store.

    The retriever will:
    - Take a user question (string)
    - Embed it using the same embeddings
    - Perform similarity search in the FAISS index
    - Return the top-k most relevant documents

    Parameters
    ----------
    vector_store : FAISS
        The FAISS vector store previously built from the transcript.
    k : int
        How many chunks to return for each query (top-k).

    Returns
    -------
    retriever
        A LangChain retriever object, ready to plug into a RAG chain.
    """
    retriever = vector_store.as_retriever(
        search_kwargs={"k": k},  # number of chunks to retrieve per question
    )
    return retriever
