import csv
from datetime import datetime
from itertools import product
import json
import numpy as np
import os
from tqdm import tqdm

from clearml import Task

from evaluation.core.evaluators.retrieval.retrieval_metrics import methodlogy_2_retrieval_metrics


with open(os.path.join(os.path.dirname(__file__), './data/input_parameters.json'), 'r') as config_file:
    params = json.load(config_file)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
FHIR_DATA_CHUNKS_FILE = os.path.join(DATA_DIR, 'fhir', params['FHIR_DATA_CHUNKS_FILE'])
OPEN_AI_OUTPUT_DATA = os.path.join(DATA_DIR, 'openai_outputs', params['OPEN_AI_OUTPUT_DATA'])
EXPERIMENTS_OUTPUT_CSV = os.path.join(DATA_DIR, 'rag_retrieval')


if __name__ == "__main__":
    # Load data
    with open(FHIR_DATA_CHUNKS_FILE, 'r') as f:
        parsed_data = json.load(f)

    openai_responses = []
    with open(OPEN_AI_OUTPUT_DATA, 'r') as f:
        for line in f:
            openai_responses.append(json.loads(line))

    # Read input parameters
    num_sampled_questions = params['num_sampled_questions']
    endpoint_url = params['endpoint_url']
    experiment_name = params['experiment_name']
    methodology = params['methodology']
    boosting_combinations = params.get('boosting_combinations', [])

    # Create task if boosting_combinations is empty
    task = None
    if not boosting_combinations:
        unique_task_name = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task = Task.init(project_name="Fasten", task_name=unique_task_name)
        task.connect(params)

    if boosting_combinations:
        # Define the range for text and embedding boosts
        boost_values = np.arange(boosting_combinations[0], boosting_combinations[1] + 0.25, 0.25)

        # Calculate all combinations of text_boost and embedding_boost
        combinations = [(text_boost, embedding_boost)
                        for text_boost, embedding_boost in product(boost_values, boost_values)
                        if not (text_boost == 0 and embedding_boost == 0)]

        # Prepare CSV file to store results
        csv_file = f"{EXPERIMENTS_OUTPUT_CSV}/metrics_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_columns = ['embedding_boost', 'text_boost', 'Retrieval Accuracy', 'Average Position',
                       'MRR', 'Average Precision', 'Average Recall', 'Total Questions',
                       'Total Openai json error', 'Total contexts found', 'Total positions sum']

        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()

            # Use tqdm to iterate over all combinations of text_boost and embedding_boost
            for text_boost, embedding_boost in tqdm(combinations, desc="Processing combinations"):
                id, counts = np.unique([resource["resource_id"] for resource in parsed_data["entry"]],
                                        return_counts=True)
                resources_counts = dict(zip(id, counts))
                metrics = methodlogy_2_retrieval_metrics(resources_counts,
                                                         openai_responses,
                                                         num_sampled_questions,
                                                         endpoint_url,
                                                         text_boost,
                                                         embedding_boost)

                # Add boost values to metrics
                metrics['embedding_boost'] = embedding_boost
                metrics['text_boost'] = text_boost

                # Write metrics to CSV
                writer.writerow(metrics)

        print(f"Metrics saved to {csv_file}")
    else:
        id, counts = np.unique([resource["resource_id"] for resource in parsed_data["entry"]],
                               return_counts=True)
        resources_counts = dict(zip(id, counts))
        metrics = methodlogy_2_retrieval_metrics(resources_counts,
                                                 openai_responses,
                                                 num_sampled_questions,
                                                 endpoint_url,
                                                 params['search_strategy']["text_boost"],
                                                 params['search_strategy']["embedding_boost"])

        # Upload metrics to ClearML
        for series_name, value in metrics.items():
            task.get_logger().report_single_value(name=series_name, value=value)

    # Close the task if created
    if task:
        task.close()
