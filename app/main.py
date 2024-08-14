import json
import os

from elasticsearch import helpers
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from .db.index_documents import bulk_load_from_json_flattened_file
from .config.elasticsearch_config import create_index_if_not_exists
from .services.search_documents import search_query
from .services.process_search_output import process_search_output, \
    stream_llm_response
from .models.sentence_transformer import get_sentence_transformer
from .config.settings import settings, logger


app = FastAPI()

os.makedirs(settings.upload_dir, exist_ok=True)

# Initialize SentenceTransformer model and elastic client
embedding_model = get_sentence_transformer()

# Create elasticsearch index
es_client = create_index_if_not_exists(settings.index_name)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(settings.upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    index_pdf(file_path, embedding_model, es_client)
    logger.info(f"File uploaded: {file.filename}")
    return {"filename": file.filename}


@app.post("/bulk_load")
async def bulk_load(file: UploadFile = File(...)):
    data = await file.read()
    # json to dict
    json_data = json.loads(data)
    # Bulk load to Elasticsearch
    try:
        helpers.bulk(es_client,
                     bulk_load_from_json_flattened_file(json_data,
                                                        embedding_model,
                                                        settings.index_name))
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


@app.get("/search")
async def search_documents(query: str, k: int = 5, threshold: float = 0):
    results = search_query(query, embedding_model, es_client, k=k,
                           threshold=threshold)
    return results


@app.get("/generate")
async def answer_query(query: str, k: int = 5, threshold: float = 0):
    results = search_query(query, embedding_model, es_client, k=k,
                           threshold=threshold)
    if not results:
        concatenated_content = "There is no context"
    else:
        concatenated_content = process_search_output(results)
    concatenated_content = process_search_output(results)
    return stream_llm_response(concatenated_content, query)
