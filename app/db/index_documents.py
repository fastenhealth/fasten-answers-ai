import os
import json

import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from ..config.settings import settings, logger


text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
)

json_text_splitter = RecursiveCharacterTextSplitter(
    separators=[',', ':', ' ', ''],
    chunk_size=600,
    chunk_overlap=50,
    length_function=len
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


def create_documents_from_json(data: dict,
                               text_splitter=json_text_splitter):
    documents = []

    for entry in data.get("entry", []):
        if "resource" in entry:
            resource = entry["resource"]
            resource_type = resource.get("resourceType")
            resource_id = resource.get("id")

            entry_text = json.dumps(resource)

            chunks = text_splitter.split_text(entry_text)
            for chunk in chunks:
                documents.append(Document(
                    text=chunk.replace('\"', '').replace('\\', ''),
                    metadata={"resourceType": resource_type,
                              "resource_id": resource_id}
                ))
    return documents


def bulk_load_from_lamaindex_documents(documents,
                                       embedding_model,
                                       index_name):
    """
    Function to load in bulk mode a JSON file FHIR for first time
    """
    for document in documents:
        resource_type = document.metadata("resourceType")
        resource_id = document.metadata("resource_id")
        document_text = document.text
        document_embedding = embedding_model.encode(document_text)
        yield {
            "_index": index_name,
            "_source": {
                "content": document_text,
                "embedding": document_embedding,
                "metadata": {
                    "resourceType": resource_type,
                    "resource_id": resource_id
                }
            }
        }


def bulk_load_from_json_file(data: dict,
                             embedding_model,
                             index_name):
    """
    Function to load in bulk mode a JSON file FHIR for first time
    """
    for entry in data.get("entry", []):
        if "resource" in entry:
            resource_type = entry.get("resourceType")
            resource_id = entry.get("resource_id")
            node_text = entry.get("resource")
            node_embedding = embedding_model.encode(node_text)
            yield {
                "_index": index_name,
                "_source": {
                    "content": node_text,
                    "embedding": node_embedding,
                    "metadata": {
                        "resourceType": resource_type,
                        "resource_id": resource_id
                    }
                }
            }
