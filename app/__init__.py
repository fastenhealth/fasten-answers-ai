from fastapi import FastAPI
from app.config.elasticsearch_config import create_index_if_not_exists
from app.models.sentence_transformer import get_sentence_transformer
from app.config.settings import settings
from app.services.reranking import RerankingService


embedding_model = get_sentence_transformer()
es_client = create_index_if_not_exists(settings.elasticsearch.index_name)
reranker_service = RerankingService()


def create_app():
    app = FastAPI()

    from app.routes.database_endpoints import router as database_router
    from app.routes.llm_endpoints import router as llm_router
    from app.routes.openai_endpoints import router as openai_router
    from app.routes.evaluation_endpoints import router as evaluation_router

    app.include_router(database_router, prefix="/database")
    app.include_router(llm_router, prefix="/generation")
    app.include_router(openai_router, prefix="/openai")
    app.include_router(evaluation_router, prefix="/evaluation")

    return app


app = create_app()
