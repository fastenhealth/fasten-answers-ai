import os

import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..config.settings import settings, logger


text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
)


def extract_text_from_pdf(pdf_path):
    logger.info(f"Extracting text from PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def index_pdf(pdf_path, embedding_model,
              es_client, index_name=settings.index_name):
    logger.info(f"Indexing PDF: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    chunks = text_splitter.create_documents(
        texts=[text],
        metadatas=[{"source": os.path.basename(pdf_path)}]
    )
    for chunk in chunks:
        content = chunk.page_content
        metadata = chunk.metadata
        embedding = embedding_model.encode(content).tolist()
        es_client.index(
            index=index_name,
            body={
                "content": content,
                "embedding": embedding,
                "metadata": metadata
            }
        )
        logger.info(f"Chunk indexed: {content[:30]}...")
    es_client.indices.refresh(index=index_name)
    logger.info(f"PDF {os.path.basename(pdf_path)} indexed successfully.")
