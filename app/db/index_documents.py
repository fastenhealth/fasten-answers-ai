import os
import json

import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from ..config.settings import settings, logger


def bulk_load_from_json_flattened_file(data: dict,
                                       embedding_model,
                                       index_name):
    """
    Function to load in bulk mode a FHIR JSON flattened file
    """
    for _, value in data.items():
        file_name = value.get("file_name")
        text_chunk = value.get("text_chunk")
        embedding = embedding_model.encode(text_chunk)
        yield {
            "_index": index_name,
            "_source": {
                "content": text_chunk,
                "embedding": embedding,
                "metadata": {
                    "file_name": file_name
                }
            }
        }
