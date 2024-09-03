from datetime import datetime
import os
import json
import pandas as pd

from clearml import Task
from dotenv import load_dotenv

from evaluation.core.evaluators.generation.correctness import CorrectnessEvaluator
from evaluation.core.evaluators.generation.faithfullness import FaithfulnessEvaluator


# Load env variables
load_dotenv()

# Load config
INPUT_PATH = os.path.join(os.path.dirname(
    __file__), "data", "input_get_metrics.json")
with open(INPUT_PATH, "r") as config_file:
    config = json.load(config_file)

DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data"))

# Files
REFERENCE_ANSWERS_FILE = os.path.join(
    DATA_DIR, "openai_outputs", config["REFERENCE_ANSWERS_FILE"])
GENERATED_ANSWERS_FILE = os.path.join(
    DATA_DIR, "rag_generation", config["GENERATED_ANSWERS_FILE"])

# Openai API Key and model to use
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = config.get("LLM_MODEL")

# Output files
CORRECTNESS_RESULTS_CSV = os.path.join(
    DATA_DIR, "openai_outputs", config["CORRECTNESS_RESULTS_CSV"])
FAITHFULNESS_RESULTS_CSV = os.path.join(
    DATA_DIR, "openai_outputs", config["FAITHFULNESS_RESULTS_CSV"])
RESULT_METRICS_TXT = os.path.join(os.path.dirname(
    __file__), "data", "output_generation_metrics.txt")

# Columns of interest
QUERY_COLUMN = config["QUERY_COLUMN"]
REFERENCE_ANSWER_COLUMN = config["REFERENCE_ANSWER_COLUMN"]
GENERATED_ANSWER_COLUMN = config["GENERATED_ANSWER_COLUMN"]
CONTEXTS_COLUMN = config["CONTEXTS_COLUMN"]
RESOURCE_ID_COLUMN = config["RESOURCE_ID_COLUMN"]

# Experiment name
EXPERIMENT_NAME = config["EXPERIMENT_NAME"]
UPLOAD_EXPERIMENT = config["UPLOAD_EXPERIMENT"]
UPLOAD_ARTIFACTS = config["UPLOAD_ARTIFACTS"]


def main():
    # Create experiment if needed
    if UPLOAD_EXPERIMENT:
        unique_task_name = f"{EXPERIMENT_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task = Task.init(project_name="Fasten", task_name=unique_task_name)
        task.connect(config)

    # Read data
    generated_answers = pd.read_csv(GENERATED_ANSWERS_FILE)
    reference_answers = pd.read_csv(REFERENCE_ANSWERS_FILE)

    generated_answers[REFERENCE_ANSWER_COLUMN] = reference_answers[REFERENCE_ANSWER_COLUMN]
    generated_answers[RESOURCE_ID_COLUMN] = reference_answers[RESOURCE_ID_COLUMN]

    # Instantiate evaluators
    correctness_evaluator = CorrectnessEvaluator(OPENAI_API_KEY,
                                                 LLM_MODEL,
                                                 threshold=4.0,
                                                 max_tokens=300)
    faithfulness_evaluator = FaithfulnessEvaluator(OPENAI_API_KEY,
                                                   LLM_MODEL,
                                                   max_tokens=300)
    # Run batch evaluations
    correctnes_mean_score = correctness_evaluator.run_batch_evaluation(
        generated_answers,
        CORRECTNESS_RESULTS_CSV,
        QUERY_COLUMN,
        REFERENCE_ANSWER_COLUMN,
        GENERATED_ANSWER_COLUMN,
        RESOURCE_ID_COLUMN
    )

    faithfulness_relevancy, \
        faithfulness_accuracy, \
        faithfulness_conciseness_and_pertinence = faithfulness_evaluator.run_batch_evaluation(
            generated_answers,
            FAITHFULNESS_RESULTS_CSV,
            GENERATED_ANSWER_COLUMN,
            CONTEXTS_COLUMN,
            RESOURCE_ID_COLUMN
        )

    metrics = {"Correctness mean score": correctnes_mean_score,
               "Faithfulness relevancy": faithfulness_relevancy,
               "Faithfulness accuracy": faithfulness_accuracy,
               "Faithfulness conciseness and pertinence": faithfulness_conciseness_and_pertinence}

    # Save metrics to txt
    with open(RESULT_METRICS_TXT, 'w') as f:
        f.write(f"Correctness score: {correctnes_mean_score}\n\n")
        f.write(f"Faithfulness relevancy score: {faithfulness_relevancy}\n")
        f.write(f"Faithfulness accuracy score: {faithfulness_accuracy}\n")
        f.write(
            f"Faithfulness conciseness_and_pertinence score: {faithfulness_conciseness_and_pertinence}\n")

    # Upload metrics and artifacts to ClearML
    if task:
        for metric_name, value in metrics.items():
            task.get_logger().report_single_value(name=metric_name, value=value)

        artifact_files = {
            "Correctness Results": CORRECTNESS_RESULTS_CSV,
            "Faithfulness Results": FAITHFULNESS_RESULTS_CSV,
        }

        if UPLOAD_ARTIFACTS:
            for artifact_name, file_path in artifact_files.items():
                task.upload_artifact(name=artifact_name,
                                     artifact_object=file_path)

        task.close()


if __name__ == "__main__":
    main()
