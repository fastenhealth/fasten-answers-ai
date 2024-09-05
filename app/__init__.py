from fastapi import FastAPI
from config.elasticsearch_config import create_index_if_not_exists
from models.sentence_transformer import get_sentence_transformer
from config.settings import settings


app = FastAPI()

embedding_model = get_sentence_transformer()

es_client = create_index_if_not_exists(settings.elasticsearch.index_name)


def create_app():
    from routes.database_endpoints import router as database_router
    from routes.llm_endpoints import router as llm_router
    from routes.processing_endpoints import router as processing_router

    app.include_router(database_router, prefix="/database")
    app.include_router(llm_router, prefix="/generation")
    app.include_router(processing_router, prefix="/processing")

    return app
