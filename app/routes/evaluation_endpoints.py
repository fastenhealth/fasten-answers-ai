import json
import numpy as np
from datetime import datetime
import os
import pandas as pd

from clearml import Task
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from app import es_client, embedding_model
from app.config.settings import logger, settings
from app.evaluation.retrieval.retrieval_metrics import evaluate_resources_summaries_retrieval
from app.evaluation.generation.correctness import CorrectnessEvaluator
from app.evaluation.generation.faithfulness import FaithfulnessEvaluator
from app.processor.files_processor import ensure_data_directory_exists, \
    generate_output_filename
from app.processor.openai_processor import jsonl_dataset_to_dataframe
from app.services.search_documents import fetch_all_documents
from app.services.conversation import batch_generation_synchronous


router = APIRouter()


@router.post("/evaluate_retrieval")
async def evaluate_retrieval(
    file: UploadFile = File(...),
    index_name: str = Form(settings.elasticsearch.index_name),
    size: int = Form(2000),
    search_text_boost: float = Form(1),
    search_embedding_boost: float = Form(1),
    k: int = Form(5),
    rerank_top_k: int = Form(0),
    urls_in_resources: bool = Form(None),
    questions_with_ids_and_dates: str = Form(None),
    chunk_size: int = Form(None),
    chunk_overlap: int = Form(None),
    clearml_track_experiment: bool = Form(False),
    clearml_experiment_name: str = Form("Retrieval evaluation"),
    clearml_project_name: str = Form("Fasten"),
):
    # Read and process reference questions and answers in JSONL
    try:
        qa_references = []

        file_data = await file.read()

        for line in file_data.decode("utf-8").splitlines():
            qa_references.append(json.loads(line))
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON format.")
    # Count total chunks by resource in database
    try:
        documents = fetch_all_documents(
            index_name=index_name, es_client=es_client, size=size)
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
                "rerank_top_k": rerank_top_k,
                "urls_in_resources": urls_in_resources,
                "questions_with_ids_and_dates": questions_with_ids_and_dates,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
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
            k=k,
            rerank_top_k=rerank_top_k,
        )

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


@router.post("/evaluate_generation")
async def evaluate_generation(
    file: UploadFile = File(...),
    openai_api_key: str = Form(...),
    openai_model: str = Form(
        "gpt-4o-mini-2024-07-18"),
    max_tokens: int = Form(400),
    limit: int = Form(None),
    # Can be 'correctness' or 'faithfulness'
    evaluation_type: str = Form(...),
    query_column: str = Form("openai_query"),
    reference_answer_column: str = Form(
        "openai_answer"),
    generated_answer_column: str = Form("response"),
    resource_id_column: str = Form(
        "resource_id_source"),
    contexts_column: str = Form("context"),
    correctness_threshold: float = Form(4.0),
    process: str = Form("openai_response"),
    job: str = Form("evaluation_correctness"),
    # Experiment parameters
    search_text_boost: float = Form(1),
    search_embedding_boost: float = Form(1),
    k: int = Form(5),
    urls_in_resources: bool = Form(None),
    questions_with_ids_and_dates: str = Form(None),
    chunk_size: int = Form(None),
    chunk_overlap: int = Form(None),
    clearml_track_experiment: bool = Form(False),
    clearml_experiment_name: str = Form("Evaluation"),
        clearml_project_name: str = Form("Fasten")):
    try:
        # Read csv
        data = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(data))
        
        if limit:
            df = df.iloc[:limit, :]

        # Check columns
        if evaluation_type == "correctness":
            required_columns = [query_column, reference_answer_column,
                                generated_answer_column, resource_id_column]
        elif evaluation_type == "faithfulness":
            required_columns = [generated_answer_column,
                                contexts_column, resource_id_column]
        else:
            raise HTTPException(
                status_code=400, detail="Invalid evaluation type. Must be 'correctness' or 'faithfulness'.")

        missing_columns = [
            col for col in required_columns if col not in df.columns]
        if missing_columns:
            return {"error": f"Missing required columns: {', '.join(missing_columns)}"}

        # Create output directory
        data_dir = ensure_data_directory_exists()
        output_file = os.path.join(
            data_dir, generate_output_filename(process=process, task=job))

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error processing the file: {str(e)}")

    # Evaluate generation
    try:
        if clearml_track_experiment:
            params = {
                "search_text_boost": search_text_boost,
                "search_embedding_boost": search_embedding_boost,
                "k": k,
                "urls_in_resources": urls_in_resources,
                "questions_with_ids_and_dates": questions_with_ids_and_dates,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "openai_max_tokens": max_tokens,
                "openai_model": openai_model
            }
            unique_task_name = f"{clearml_experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            task = Task.init(project_name=clearml_project_name,
                             task_name=unique_task_name)
            task.connect(params)

        if evaluation_type == "correctness":
            evaluator = CorrectnessEvaluator(
                openai_api_key, openai_model, threshold=correctness_threshold, max_tokens=max_tokens)
            evaluation_metrics = evaluator.run_batch_evaluation(df=df, output_file=output_file,
                                                                query_column=query_column,
                                                                reference_answer_column=reference_answer_column,
                                                                generated_answer_column=generated_answer_column,
                                                                resource_id_column=resource_id_column)
        elif evaluation_type == "faithfulness":
            evaluator = FaithfulnessEvaluator(
                openai_api_key, openai_model, max_tokens=max_tokens)
            evaluation_metrics = evaluator.run_batch_evaluation(df=df, output_file=output_file,
                                                                generated_answer_column=generated_answer_column,
                                                                contexts_column=contexts_column,
                                                                resource_id_column=resource_id_column)

        # Upload metrics to clearml
        if clearml_track_experiment and task:
            for series_name, value in evaluation_metrics.items():
                task.get_logger().report_single_value(name=series_name, value=value)
            task.close()

        return evaluation_metrics

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error during {evaluation_type} evaluation: {str(e)}")


@router.post("/batch_generation")
async def batch_generation(file: UploadFile = File(...),
                           limit: int = File(None),
                           question_column: str = File("openai_query"),
                           model_prompt: str = Form("llama3.1"),
                           llm_model: str = Form("llama3.1"),
                           search_text_boost: float = Form(1),
                           search_embedding_boost: float = Form(1),
                           k: int = Form(5),
                           process: str = Form("local_llm_response"),
                           job: str = Form("generation_evaluation")):
    # Openai batch response in JSONL to dataframe
    try:
        data = await file.read()

        if file.filename.endswith(".jsonl"):
            df = jsonl_dataset_to_dataframe(data)
            if limit:
                df = df.iloc[:limit, :]
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Only JSONL from Openai batch api supported.")

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid input file format.")
    # Generate responses
    try:
        # Select model prompt
        model_prompt = settings.model.conversation_model_prompt.get(
            model_prompt)

        # Do queries and generations
        output_file = batch_generation_synchronous(
            model_prompt=model_prompt,
            es_client=es_client,
            embedding_model=embedding_model,
            input_data=df,
            question_column=question_column,
            k=k,
            text_boost=search_text_boost,
            embedding_boost=search_embedding_boost,
            llm_model=llm_model,
            process=process,
            job=job
        )
        return {"Detail": "Batch generated successfully.",
                "Output file": output_file}

    except Exception as e:
        logger.error(f"Error during batch generation: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error during batch generation: {str(e)}")
