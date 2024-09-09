import json
import numpy as np
from datetime import datetime

from clearml import Task
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from app import es_client, embedding_model
from app.config.settings import logger, settings
from app.evaluation.retrieval.retrieval_metrics import evaluate_resources_summaries_retrieval
from app.services.search_documents import fetch_all_documents


router = APIRouter()


@router.post("/evaluate_retrieval")
async def evaluate_retrieval(file: UploadFile = File(...),
                             index_name: str = Form(
                                 settings.elasticsearch.index_name),
                             size: int = Form(2000),
                             search_text_boost: float = Form(1),
                             search_embedding_boost: float = Form(1),
                             k: int = Form(5),
                             urls_in_resources: bool = Form(None),
                             questions_with_ids_and_dates: str = Form(None),
                             chunk_size: int = Form(None),
                             chunk_overlap: int = Form(None),
                             clearml_track_experiment: bool = Form(False),
                             clearml_experiment_name: str = Form("Retrieval evaluation"),
                             clearml_project_name: str = Form("Fasten")):
    # Read and process reference questions and answers in JSONL
    try:
        qa_references = []

        file_data = await file.read()

        for line in file_data.decode('utf-8').splitlines():
            qa_references.append(json.loads(line))
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON format.")
    # Count total chunks by resource in database
    try:
        documents = fetch_all_documents(
            index_name=index_name,
            es_client=es_client,
            size=size)
        id, counts = np.unique([resource["metadata"]["resource_id"]
                               for resource in documents], return_counts=True)
        resources_counts = dict(zip(id, counts))
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error retrieving documents: {str(e)}")
    # Evaluate retrieval
    try:
        if clearml_track_experiment:
            params = {
                "search_text_boost": search_text_boost,
                "search_embedding_boost": search_embedding_boost,
                "k": k,
                "urls_in_resources": urls_in_resources,
                "questions_with_ids_and_dates": questions_with_ids_and_dates,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
            unique_task_name = f"{clearml_experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            task = Task.init(project_name=clearml_project_name,
                             task_name=unique_task_name)
            task.connect(params)

        retrieval_metrics = evaluate_resources_summaries_retrieval(
            es_client=es_client,
            embedding_model=embedding_model,
            resource_chunk_counts=resources_counts,
            qa_references=qa_references,
            search_text_boost=search_text_boost,
            search_embedding_boost=search_embedding_boost,
            k=k)

        # Upload metrics and close task
        if task:
            for series_name, value in retrieval_metrics.items():
                task.get_logger().report_single_value(name=series_name, value=value)

            task.close()

        return retrieval_metrics

    except Exception as e:
        logger.error(f"Error during retrieval evaluation: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error during retrieval evaluation: {str(e)}")
