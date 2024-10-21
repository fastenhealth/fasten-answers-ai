import os

from app.services.llama_client import llm_client
from evaluation.evaluation_metrics.evaluate_generation.generation_speed.utils.parameters import load_params
from evaluation.evaluation_metrics.evaluate_generation.generation_speed.utils.settings import MODEL_SETTINGS

from evaluation.evaluation_metrics.evaluate_generation.generation_speed.utils.generator import generate_responses
from evaluation.evaluation_metrics.evaluate_generation.generation_speed.utils.contexts_and_questions import (
    create_contexts_and_questions,
)


def main():
    current_dir = os.path.dirname(__file__)
    input_path = os.path.abspath(os.path.join(current_dir, "data", "input.json"))
    params = load_params(input_path)
    model_name = params.get("model")
    model_prompt = MODEL_SETTINGS[model_name]["model_prompt"]

    df_contexts_and_questions = create_contexts_and_questions(MODEL_SETTINGS[model_name]["tokenizer"])

    generate_responses(params, df_contexts_and_questions, llm_client, model_prompt)


if __name__ == "__main__":
    main()
