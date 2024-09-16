import os
import json
import pandas as pd

from evaluation.core.evaluators.generation.correctness import CORRECTNESS_SYS_TMPL, CORRECTNESS_USER_TMPL
from evaluation.core.evaluators.generation.faithfullness import FAITHFULLNESS_SYS_TMPL, FAITHFULLNESS_USER_TMPL
from evaluation.core.openai.openai import calculate_total_tokens, calculate_api_costs


# Load config
INPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "input_estimate_costs.json")

with open(INPUT_PATH, "r") as config_file:
    config = json.load(config_file)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))

# Files
REFERENCE_ANSWERS_FILE = os.path.join(DATA_DIR, "openai_outputs", config["REFERENCE_ANSWERS_FILE"])
GENERATED_ANSWERS_FILE = os.path.join(DATA_DIR, "rag_generation", config["GENERATED_ANSWERS_FILE"])

# Interest columns
QUERY_COLUMN = config.get("QUERY_COLUMN")
CONTEXTS_COLUMN = config.get("CONTEXTS_COLUMN")
GENERATED_ANSWER_COLUMN = config.get("GENERATED_ANSWER_COLUMN")
REFERENCE_ANSWER_COLUMN = config.get("REFERENCE_ANSWER_COLUMN")

# Output costs
OUTPUT_COSTS = os.path.join(os.path.dirname(__file__), "data", "output_estimate_costs.txt")
COST_PER_MILLION_INPUT_TOKENS = config.get("COST_PER_MILLION_INPUT_TOKENS")
COST_PER_MILLION_OUTPUT_TOKENS = config.get("COST_PER_MILLION_OUTPUT_TOKENS")


def main():
    # Read data
    generated_answers = pd.read_csv(GENERATED_ANSWERS_FILE)
    reference_answers = pd.read_csv(REFERENCE_ANSWERS_FILE)

    generated_answers[REFERENCE_ANSWER_COLUMN] = reference_answers[REFERENCE_ANSWER_COLUMN]
    # Calculate total tokens
    total_tokens_input_correctness = calculate_total_tokens(
        df=generated_answers,
        query_column=QUERY_COLUMN,
        generated_answer_column=GENERATED_ANSWER_COLUMN,
        contexts_column=CONTEXTS_COLUMN,
        reference_answer_column=REFERENCE_ANSWER_COLUMN,
        system_template=CORRECTNESS_SYS_TMPL,
        user_template=CORRECTNESS_USER_TMPL,
        template_type="correctness",
        encoding_name="o200k_base",
    )
    total_tokens_input_faithfulness = calculate_total_tokens(
        df=generated_answers,
        query_column=QUERY_COLUMN,
        generated_answer_column=GENERATED_ANSWER_COLUMN,
        contexts_column=CONTEXTS_COLUMN,
        reference_answer_column=REFERENCE_ANSWER_COLUMN,
        system_template=FAITHFULLNESS_SYS_TMPL,
        user_template=FAITHFULLNESS_USER_TMPL,
        template_type="faithfulness",
        encoding_name="o200k_base",
    )

    # Calculate costs
    total_openai_queries = len(reference_answers)

    cost_estimate_correctness = calculate_api_costs(
        total_input_tokens=total_tokens_input_correctness,
        total_openai_requests=total_openai_queries,
        cost_per_million_input_tokens=COST_PER_MILLION_INPUT_TOKENS,
        cost_per_million_output_tokens=COST_PER_MILLION_OUTPUT_TOKENS,
        tokens_per_response=300,
    )
    cost_estimate_faithfulness = calculate_api_costs(
        total_input_tokens=total_tokens_input_faithfulness,
        total_openai_requests=total_openai_queries,
        cost_per_million_input_tokens=COST_PER_MILLION_INPUT_TOKENS,
        cost_per_million_output_tokens=COST_PER_MILLION_OUTPUT_TOKENS,
        tokens_per_response=300,
    )

    # Save total costs
    with open(OUTPUT_COSTS, "w") as f:
        f.write("Total Estimated Costs:\n")
        f.write(f"${cost_estimate_correctness['total_cost'] + cost_estimate_faithfulness['total_cost']}\n\n\n")
        f.write("Correctness Evaluation Costs:\n")
        f.write(f"Estimated Total Costs: ${cost_estimate_correctness['total_cost']}\n")
        f.write(f"Input Costs: ${cost_estimate_correctness['input_cost']}\n")
        f.write(f"Output Costs: ${cost_estimate_correctness['output_cost']}\n")
        f.write(f"Total Input Tokens: {cost_estimate_correctness['total_input_tokens']}\n")
        f.write(f"Total Output Tokens: {cost_estimate_correctness['total_output_tokens']}\n\n")

        f.write("\nFaithfulness Evaluation Costs:\n")
        f.write(f"Estimated Total Costs: ${cost_estimate_faithfulness['total_cost']}\n")
        f.write(f"Input Costs: ${cost_estimate_faithfulness['input_cost']}\n")
        f.write(f"Output Costs: ${cost_estimate_faithfulness['output_cost']}\n")
        f.write(f"Total Input Tokens: {cost_estimate_faithfulness['total_input_tokens']}\n")
        f.write(f"Total Output Tokens: {cost_estimate_faithfulness['total_output_tokens']}\n")


if __name__ == "__main__":
    main()
