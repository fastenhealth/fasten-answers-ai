import os

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from .db.index_documents import index_pdf
from .config.elasticsearch_config import get_es_client
from .services.search_documents import search_query
from .services.process_search_output import process_search_output, \
      stream_llm_response
from .models.sentence_transformer import get_sentence_transformer
from .config.settings import settings, logger


app = FastAPI()

os.makedirs(settings.upload_dir, exist_ok=True)

# Initialize SentenceTransformer model and elastic client
embedding_model = get_sentence_transformer()
es_client = get_es_client()


def create_index_if_not_exists(index_name):
    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name)
        logger.info(f"Index '{index_name}' created.")


create_index_if_not_exists(settings.index_name)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(settings.upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    index_pdf(file_path, embedding_model, es_client)
    logger.info(f"File uploaded: {file.filename}")
    return {"filename": file.filename}


@app.get("/search")
async def search_documents(query: str, k: int = 5, threshold: float = 0):
    results = search_query(query, embedding_model, es_client, k=k,
                           threshold=threshold)
    return JSONResponse(content=results)


@app.get("/generate")
async def answer_query(query: str, k: int = 5, threshold: float = 0):
    results = search_query(query, embedding_model, es_client, k=k,
                           threshold=threshold)
    if not results:
        concatenated_content = f"No results found for query: {query}"
    else:
        concatenated_content = process_search_output(results)
    concatenated_content = process_search_output(results)
    return stream_llm_response(concatenated_content, query)
