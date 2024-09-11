import json

from fastapi import APIRouter, HTTPException, UploadFile, \
    File, Form, status

from app import es_client, embedding_model
from app.config.settings import settings
from app.processor.fhir_processor import process_resources
from app.services.conversation import process_search_output, llm_response
from app.services.search_documents import search_query
from app.services.summarize import summarize_resources_parallel


router = APIRouter()


@router.get("/generate")
async def answer_query(
    query: str, k: int = 5, params=None, stream: bool = False, text_boost: float = 0.25, embedding_boost: float = 4.0
):
    results = search_query(query, embedding_model, es_client, k=k,
                           text_boost=text_boost, embedding_boost=embedding_boost)
    if not results:
        concatenated_content = "There is no context"
    else:
        concatenated_content, resources_id = process_search_output(results)

    return llm_response(concatenated_content, query, resources_id, stream, params)


@router.post("/summarize_and_load_parallel")
async def summarize_and_load(
    file: UploadFile = File(...),
    remove_urls: bool = Form(True),
    batch_size: int = Form(4),
    limit: int = Form(None),
):
    # Read file
    try:
        resources = await file.read()
        resources = json.loads(resources)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON format.")
    # Process input file
    try:
        limit = 1 if limit <= 0 else limit
        if limit:
            resources_processed = process_resources(
                data=resources, remove_urls=remove_urls)[:limit]
        else:
            resources_processed = process_resources(
                data=resources, remove_urls=remove_urls)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error during processing: {str(e)}")
    # Generate summaries and save
    try:
        output_file = await summarize_resources_parallel(model_prompt=settings.model.summaries_model_prompt,
                                                         es_client=es_client,
                                                         embedding_model=embedding_model,
                                                         resources=resources_processed,
                                                         batch_size=batch_size)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error during summaries generation: {str(e)}")

    return {"detail": "Data summarized and loaded successfully.", "Output file": output_file}
