import os
import json
import pandas as pd

from dotenv import load_dotenv

from evaluation.core.evaluators.generation.correctness import CorrectnessEvaluator
from evaluation.core.evaluators.generation.faithfullness import FaithfulnessEvaluator


# Load env variables
load_dotenv()

# Load config
with open(os.path.join(os.path.dirname(__file__), "/data/input_get_metrics.json"), "r") as config_file:
    config = json.load(config_file)

DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "data"))

REFERENCE_ANSWERS_FILE = os.path.join(
    DATA_DIR, "openai_outputs", config["REFERENCE_ANSWERS_FILE"])
GENERATED_ANSWERS_FILE = os.path.join(
    DATA_DIR, "rag_generation", config["GENERATED_ANSWERS_FILE"])
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Output files
CORRECTNESS_RESULTS_CSV = os.path.join(
    DATA_DIR, "openai_outputs", config["CORRECTNESS_RESULTS"])
FAITHFULLNESS_RESULTS_CSV = os.path.join(
    DATA_DIR, "openai_outputs", config["FAITHFULLNESS_RESULTS"])
RESULT_METRICS_TXT = os.path.join(
    DATA_DIR, "rag_generation", config["RESULT_METRICS_TXT"])

# Columns of interest
QUERY_COLUMN = config["QUERY_COLUMN"]
REFERENEC_ANSWER_COLUMN = config["REFERENEC_ANSWER_COLUMN"]
GENERATED_ANSWER_COLUMN = config["GENERATED_ANSWER_COLUMN"]
CONTEXTS_COLUMN = config["CONTEXTS_COLUMN"]


def main():
    # Read data
    generated_answers = pd.read_csv(GENERATED_ANSWERS_FILE)
    reference_answers = pd.read_csv(REFERENCE_ANSWERS_FILE)

    generated_answers["reference_answer"] = reference_answers["openai_answer"]

    # Instantiate evaluators
    correctness_evaluator = CorrectnessEvaluator(OPENAI_API_KEY,
                                                 "gpt-4o-2024-08-06",
                                                 threshold=4.0,
                                                 max_tokens=300)
    faithfullness_evaluator = FaithfulnessEvaluator(OPENAI_API_KEY,
                                                    "gpt-4o-2024-08-06",
                                                    threshold=4.0,
                                                    max_tokens=300)

    correctnes_mean_score = correctness_evaluator.run_batch_evaluation(
        generated_answers,
        CORRECTNESS_RESULTS_CSV,
        QUERY_COLUMN,
        REFERENEC_ANSWER_COLUMN,
        GENERATED_ANSWER_COLUMN
    )

    faithfulness_relevancy, \
        faithfulness_accuracy, \
        faithfulness_conciseness_and_pertinence = faithfullness_evaluator.run_batch_evaluation(
            generated_answers,
            FAITHFULLNESS_RESULTS_CSV,
            GENERATED_ANSWER_COLUMN
        )

    with open(RESULT_METRICS_TXT, 'w') as f:
        f.write(f"Correctness score: {correctnes_mean_score}\n\n")
        f.write(f"Faithfulness relevancy score: {faithfulness_relevancy}\n")
        f.write(f"Faithfulness accuracy score: {faithfulness_accuracy}\n")
        f.write(f"Faithfulness conciseness_and_pertinence score: {faithfulness_conciseness_and_pertinence}\n")

if __name__ == "__main__":
    main()
