from datetime import datetime
import json
import numpy as np
import csv
from tqdm import tqdm
from itertools import product

from evaluate_retrieval import methodlogy_1_retrieval_metrics, methodlogy_2_retrieval_metrics


if __name__ == "__main__":
    # Read input parameters
    with open('./data/parameters.json', 'r') as f:
        params = json.load(f)

    num_sampled_questions = params['num_sampled_questions']
    endpoint_url = params['endpoint_url']
    experiment_name = params['experiment_name']
    methodology = params['methodology']
    fhir_raw = params['data']["fhir_raw"]
    fhir_data = params['data']["fhir_data_chunks"]
    openai_input_data = params['data']["openai_input"]
    openai_output_data = params['data']["openai_output"]
    boosting_range = params.get('boosting_combinations', [0, 16])

    # Define the range for text and embedding boosts
    boost_values = np.arange(boosting_range[0], boosting_range[1] + 0.25, 0.25)

    # Calculate all combinations of text_boost and embedding_boost
    combinations = [(text_boost, embedding_boost)
                    for text_boost, embedding_boost in product(boost_values, boost_values)
                    if not (text_boost == 0 and embedding_boost == 0)]

    # Load JSON data
    with open(fhir_data, 'r') as f:
        parsed_data = json.load(f)

    openai_responses = []
    with open(openai_output_data, 'r') as f:
        for line in f:
            openai_responses.append(json.loads(line))

    # Prepare CSV file to store results
    csv_file = f"./data/output/metrics_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_columns = ['embedding_boost', 'text_boost', 'Retrieval Accuracy', 'Average Position',
                   'MRR', 'Average Precision', 'Average Recall', 'Total Questions',
                   'Total Openai json error', 'Total contexts found', 'Total positions sum']

    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()

        # Use tqdm to iterate over all combinations of text_boost and embedding_boost
        for text_boost, embedding_boost in tqdm(combinations, desc="Processing combinations"):
            # Evaluate metrics for the current combination of boost values
            if methodology == "methodology_1":
                entry_dict = {key: value["text_chunk"] for key, value in parsed_data.items()}
                metrics = methodlogy_1_retrieval_metrics(entry_dict,
                                                         openai_responses,
                                                         num_sampled_questions,
                                                         endpoint_url,
                                                         text_boost,
                                                         embedding_boost)
            elif methodology == "methodology_2":
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
