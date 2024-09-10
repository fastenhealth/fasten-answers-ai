import json

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status

from app.processor.fhir_processor import process_resources
from app.processor.openai_processor import calculate_costs, process_prompts_and_save_responses
from app.config.settings import settings


router = APIRouter()


@router.post("/execute_batch_chat_requests")
async def execute_batch_chat_requests(
    openai_api_key: str = Form(...),
    task: str = Form("summarize"),
    remove_urls: bool = Form(True),
    get_costs: bool = Form(True),
    cost_per_million_input_tokens: float = Form(0.150),
    cost_per_million_output_tokens: float = Form(0.600),
    max_tokens_per_response: int = Form(300),
    openai_model: str = Form("gpt-4o-mini-2024-07-18"),
    file: UploadFile = File(...),
):
    # Read file
    try:
        resources = await file.read()
        resources = json.loads(resources)
    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON format.")

    # Process resources
    try:
        resources_processed = process_resources(data=resources, remove_urls=remove_urls)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error during processing: {str(e)}")

    # Get model prompt for the task
    if task == "summarize":
        system_prompt = settings.model.summaries_openai_system_prompt
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported task.")

    # Calculate costs
    if get_costs:
        try:
            costs = calculate_costs(
                system_prompt=system_prompt,
                user_prompts=resources_processed,
                cost_per_million_input_tokens=cost_per_million_input_tokens,
                cost_per_million_output_tokens=cost_per_million_output_tokens,
                tokens_per_response=max_tokens_per_response,
                model=openai_model,
            )
            return costs
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error calculating costs: {str(e)}")
    # Get answers from resources
    else:
        try:
            output_file = process_prompts_and_save_responses(
                task=task,
                system_prompt=system_prompt,
                user_prompts=resources_processed,
                openai_api_key=openai_api_key,
                model=openai_model,
                max_tokens=max_tokens_per_response,
            )
            return {"message": f"Responses saved to {output_file}"}
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error generating responses: {str(e)}")
