import os
import json
import pandas as pd

from evaluation.core.evaluators.generation.correctness import CORRECTNESS_SYS_TMPL, CORRECTNESS_USER_TMPL
from evaluation.core.evaluators.generation.faithfullness import FAITHFULLNESS_SYS_TMPL, FAITHFULLNESS_USER_TMPL
from evaluation.core.openai import calculate_total_tokens, calculate_api_costs


# Load config
with open(os.path.join(os.path.dirname(__file__), "/data/input_estimate_costs.json"), "r") as config_file:
    config = json.load(config_file)

DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "data"))

REFERENCE_ANSWERS_FILE = os.path.join(
    DATA_DIR, "openai_outputs", config["REFERENCE_ANSWERS_FILE"])
GENERATED_ANSWERS_FILE = os.path.join(
    DATA_DIR, "rag_generation", config["GENERATED_ANSWERS_FILE"])

OUTPUT_COSTS = os.path.join(os.path.dirname(
    __file__), "/data/estimate_costs.txt")


def main():
    # Read data
    generated_answers = pd.read_csv(GENERATED_ANSWERS_FILE)
    reference_answers = pd.read_csv(REFERENCE_ANSWERS_FILE)

    # Calculate total tokens
    total_tokens_input_correctness = calculate_total_tokens(
        df=generated_answers,
        reference_df=reference_answers,
        system_template=CORRECTNESS_SYS_TMPL,
        user_template=CORRECTNESS_USER_TMPL,
        template_type='correctness',
        encoding_name='o200k_base'
    )
    total_tokens_input_faithfulness = calculate_total_tokens(
        df=generated_answers,
        reference_df=reference_answers,
        system_template=FAITHFULLNESS_SYS_TMPL,
        user_template=FAITHFULLNESS_USER_TMPL,
        template_type='faithfulness',
        encoding_name='o200k_base'
    )

    # Calculate costs
    total_openai_queries = len(reference_answers)

    cost_estimate_correctness = calculate_api_costs(
        total_tokens_input=total_tokens_input_correctness,
        total_openai_queries=total_openai_queries
    )
    cost_estimate_faithfulness = calculate_api_costs(
        total_tokens_input=total_tokens_input_faithfulness,
        total_openai_queries=total_openai_queries
    )

    # Save total costs
    with open(OUTPUT_COSTS, 'w') as f:
        f.write("Tota Estimated Costs:\n")
        f.write(
            f"${cost_estimate_correctness['total_costs'] + cost_estimate_faithfulness['total_costs']}\n\n\n")
        f.write("Correctness Evaluation Costs:\n")
        f.write(
            f"Estimated Total Costs: ${cost_estimate_correctness['total_costs']}\n")
        f.write(f"Input Costs: ${cost_estimate_correctness['input_costs']}\n")
        f.write(
            f"Output Costs: ${cost_estimate_correctness['output_costs']}\n")
        f.write(
            f"Total Input Tokens: {cost_estimate_correctness['total_tokens_input']}\n")
        f.write(
            f"Total Output Tokens: {cost_estimate_correctness['total_tokens_output']}\n\n")

        f.write("\nFaithfulness Evaluation Costs:\n")
        f.write(
            f"Estimated Total Costs: ${cost_estimate_faithfulness['total_costs']}\n")
        f.write(f"Input Costs: ${cost_estimate_faithfulness['input_costs']}\n")
        f.write(
            f"Output Costs: ${cost_estimate_faithfulness['output_costs']}\n")
        f.write(
            f"Total Input Tokens: {cost_estimate_faithfulness['total_tokens_input']}\n")
        f.write(
            f"Total Output Tokens: {cost_estimate_faithfulness['total_tokens_output']}\n")
