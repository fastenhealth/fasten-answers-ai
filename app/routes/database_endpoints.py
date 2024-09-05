import json

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from elasticsearch import helpers

from app import es_client, embedding_model
from config.settings import logger, settings
from db.index_documents import bulk_load_fhir_data
from services.search_documents import search_query, fetch_all_documents


router = APIRouter()


@router.post("/bulk_load")
async def bulk_load(file: UploadFile = File(...)):
    data = await file.read()
    json_data = json.loads(data)["entry"]
    try:
        helpers.bulk(es_client,
                     bulk_load_fhir_data(json_data,
                                         settings.elasticsearch.index_name))
        logger.info(f"Bulk load completed for file: {file.filename}")
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        logger.error(f"Bulk load failed: {str(e)}")
        return {"status": "error", "message": str(e)}


@router.delete("/delete_all_documents")
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


@router.get("/get_all_documents")
async def get_all_documents(index_name: str = settings.elasticsearch.index_name):
    try:
        documents = fetch_all_documents(
            index_name=index_name, es_client=es_client)
        return documents
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error retrieving documents: {str(e)}")


@router.get("/search")
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
