import json

from elasticsearch import helpers
from fastapi import FastAPI, HTTPException, UploadFile, File, status

from db.index_documents import bulk_load_fhir_data
from config.elasticsearch_config import create_index_if_not_exists
from processor.fhir_processor import process_resources
from services.search_documents import search_query, fetch_all_documents
from services.conversation import process_search_output, \
    llm_response
from services.summarize import summarize_resources
from models.sentence_transformer import get_sentence_transformer
from config.settings import settings, logger


app = FastAPI()

# Initialize SentenceTransformer model and elastic client
embedding_model = get_sentence_transformer()

# Create elasticsearch index
es_client = create_index_if_not_exists(settings.elasticsearch.index_name)


@app.post("/bulk_load")
async def bulk_load(file: UploadFile = File(...)):
    data = await file.read()
    # json to dict
    json_data = json.loads(data)["entry"]
    # Bulk load to Elasticsearch
    try:
        helpers.bulk(es_client,
                     bulk_load_fhir_data(json_data,
                                         embedding_model,
                                         settings.elasticsearch.index_name))
        logger.info(f"Bulk load completed for file: {file.filename}")
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        logger.error(f"Bulk load failed: {str(e)}")
        return {"status": "error", "message": str(e)}


@app.delete("/delete_all_documents")
async def delete_all_documents(index_name: str):
    try:

        es_client.delete_by_query(index=index_name, body={
            "query": {
                "match_all": {}
            }
        })
        logger.info(f"All documents deleted from index '{index_name}'")
        return {"status": "success",
                "message": f"All documents deleted from index '{index_name}'"}
    except Exception as e:
        logger.error(f"Failed to delete documents: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Failed to delete documents: {str(e)}")


@app.get("/get_all_documents")
async def get_all_documents(index_name: str = settings.elasticsearch.index_name):
    """
    Retrieve all documents from the specified Elasticsearch index.
    """
    try:
        documents = fetch_all_documents(index_name=index_name,
                                        es_client=es_client)
        return documents

    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error retrieving documents: {str(e)}")


@app.get("/search")
async def search_documents(query: str,
                           k: int = 5,
                           text_boost: float = 0.25,
                           embedding_boost: float = 4.0):
    results = search_query(query,
                           embedding_model,
                           es_client, k=k,
                           text_boost=text_boost,
                           embedding_boost=embedding_boost)
    return results


@app.get("/generate")
async def answer_query(query: str,
                       k: int = 5,
                       params=None,
                       stream: bool = False,
                       text_boost: float = 0.25,
                       embedding_boost: float = 4.0):
    results = search_query(query,
                           embedding_model,
                           es_client,
                           k=k,
                           text_boost=text_boost,
                           embedding_boost=embedding_boost)
    if not results:
        concatenated_content = "There is no context"
    else:
        concatenated_content, resources_id = process_search_output(results)

    return llm_response(concatenated_content, query, resources_id, stream, params)


@app.post("/summarize_and_load")
async def summarize(file: UploadFile = File(...),
                    model: str = "microsoft/Phi-3.5-mini-instruct",
                    remove_urls: bool = True,
                    stream: bool = False,
                    limit: int = None):
    # Load file
    try:
        resources = await file.read()
        resources = json.loads(resources)
    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Invalid JSON format.")
    # Process resources
    try:
        limit = 1 if limit <= 0 else limit
        if limit:
            resources_processed = process_resources(
                data=resources, remove_urls=remove_urls)[:limit]
        else:
            resources_processed = process_resources(
                data=resources, remove_urls=remove_urls)
        resources_summarized = summarize_resources(
            resources_processed, stream)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error during processing: {str(e)}")
    # Load to ElasticSearch
    try:
        helpers.bulk(es_client,
                     bulk_load_fhir_data(
                         data=resources_summarized,
                         text_key="summary",
                         embedding_model=embedding_model,
                         index_name=settings.elasticsearch.index_name
                     ))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error loading data into Elasticsearch: {str(e)}")

    return {"detail": "Data summarized and loaded successfully.",
            "resources_processed": len(resources_summarized)}


@app.get("/process_fhir")
async def process_fhir(resources: dict,
                       remove_urls: bool = True,
                       ):
    resources_processed = process_resources(data=resources,
                                            remove_urls=remove_urls)
    return resources_processed
