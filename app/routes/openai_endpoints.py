import json

from fastapi import APIRouter, HTTPException, \
    UploadFile, File, status

from processor.fhir_processor import process_resources
from processor.openai_processor import process_prompts_and_calculate_costs
from config.settings import settings


router = APIRouter()


@router.get("/execute_batch_chat_requests")
async def execute_batch_chat_requests(file: UploadFile = File(...),
                                      task: str = "summarize",
                                      remove_urls: bool = True,
                                      get_costs: bool = True,
                                      cost_per_million_input_tokens: int = 0.150,
                                      cost_per_million_output_tokens: int = 0.600,
                                      max_tokens_per_response: int = 300,
                                      openai_model: str = "gpt-4o-mini-2024-07-18"):
    # Read file
    try:
        resources = await file.read()
        resources = json.loads(resources)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON format.")
    # Process file
    try:
        resources_processed = process_resources(
            data=resources, remove_urls=remove_urls)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error during processing: {str(e)}")
    if task == "summarize":
        system_prompt = settings.model.summaries_openai_system_prompt

    if get_costs:
        costs = process_prompts_and_calculate_costs(system_prompt=system_prompt,
                                                    user_prompts=resources_processed,
                                                    cost_per_million_input_tokens=cost_per_million_input_tokens,
                                                    cost_per_million_output_tokens=cost_per_million_output_tokens,
                                                    tokens_per_response=max_tokens_per_response,
                                                    model=openai_model)
        return costs
    else:
        resources_processed = process_resources(
        data=resources, remove_urls=remove_urls)
    return resources_processed
