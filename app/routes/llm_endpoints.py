import json

from fastapi import APIRouter, HTTPException, UploadFile, File, status
from elasticsearch import helpers

from app import es_client, embedding_model
from config.settings import settings
from db.index_documents import bulk_load_fhir_data
from processor.fhir_processor import process_resources
from services.conversation import process_search_output, \
    llm_response
from services.search_documents import search_query
from services.summarize import summarize_resources


router = APIRouter()


@router.get("/generate")
async def answer_query(query: str, k: int = 5, params=None, stream: bool = False, text_boost: float = 0.25, embedding_boost: float = 4.0):
    results = search_query(query, embedding_model, es_client, k=k,
                           text_boost=text_boost, embedding_boost=embedding_boost)
    if not results:
        concatenated_content = "There is no context"
    else:
        concatenated_content, resources_id = process_search_output(results)

    return llm_response(concatenated_content, query, resources_id, stream, params)


@router.post("/summarize_and_load")
async def summarize(file: UploadFile = File(...),
                    model: str = "microsoft/Phi-3.5-mini-instruct",
                    remove_urls: bool = True,
                    stream: bool = False,
                    limit: int = None):
    try:
        resources = await file.read()
        resources = json.loads(resources)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON format.")

    try:
        limit = 1 if limit <= 0 else limit
        if limit:
            resources_processed = process_resources(
                data=resources, remove_urls=remove_urls)[:limit]
        else:
            resources_processed = process_resources(
                data=resources, remove_urls=remove_urls)
        resources_summarized = summarize_resources(resources_processed, stream)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error during processing: {str(e)}")

    try:
        helpers.bulk(es_client, bulk_load_fhir_data(data=resources_summarized, text_key="summary",
                     embedding_model=embedding_model, index_name=settings.elasticsearch.index_name))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error loading data into Elasticsearch: {str(e)}")

    return {"detail": "Data summarized and loaded successfully.", "resources_processed": len(resources_summarized)}
