from datetime import datetime
import json
import numpy as np
from clearml import Task
from evaluate_retrieval import methodlogy_1_retrieval_metrics, \
    methodlogy_2_retrieval_metrics


if __name__ == "__main__":
    # Read input parameters
    with open('./data/parameters.json', 'r') as f:
        params = json.load(f)

    num_sampled_questions = params['num_sampled_questions']
    endpoint_url = params['endpoint_url']
    experiment_name = params['experiment_name']
    methodology = params['methodology']
    search_text_boost = params['search_strategy']["text_boost"]
    search_embedding_boost = params['search_strategy']["embedding_boost"]
    fhir_raw = params['data']["fhir_raw"]
    fhir_data = params['data']["fhir_data_chunks"]
    openai_input_data = params['data']["openai_input"]
    openai_output_data = params['data']["openai_output"]

    # Initialize a new task for each execution
    unique_task_name = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    task = Task.init(project_name="Fasten", task_name=unique_task_name)

    task.connect(params)

    # Load JSON data
    with open(fhir_data, 'r') as f:
        parsed_data = json.load(f)

    openai_responses = []
    with open(openai_output_data, 'r') as f:
        for line in f:
            openai_responses.append(json.loads(line))

    # Initialize counters
    total_questions = 0
    total_contexts_found = 0
    position_sum = 0
    k = 5

    # Evaluate metrics
    if methodology == "methodology_1":
        entry_dict = {key: value["text_chunk"] for key, value in parsed_data.items()}
        metrics = methodlogy_1_retrieval_metrics(entry_dict,
                                                 openai_responses,
                                                 num_sampled_questions,
                                                 endpoint_url,
                                                 search_text_boost,
                                                 search_embedding_boost)
    elif methodology == "methodology_2":

        id, counts = np.unique([resource["resource_id"]
                                for resource in parsed_data["entry"]], return_counts=True)
        resources_counts = dict(zip(id, counts))
        metrics = methodlogy_2_retrieval_metrics(resources_counts,
                                                 openai_responses,
                                                 num_sampled_questions,
                                                 endpoint_url,
                                                 search_text_boost,
                                                 search_embedding_boost)

    # Upload metrics
    for series_name, value in metrics.items():
        task.get_logger().report_single_value(
            name=series_name,
            value=value
        )

    # Upload artifacts
    task.upload_artifact('FHIR raw', artifact_object=fhir_raw)
    task.upload_artifact('FHIR data chunks', artifact_object=fhir_data)
    task.upload_artifact('Openai input', artifact_object=openai_input_data)
    task.upload_artifact('Openai output', artifact_object=openai_output_data)

    # Close the task
    print(metrics)

    # Config file:
    config_file = './data/config.json'
    config_file = task.connect_configuration(config_file)
    my_params = json.load(open(config_file, 'rt'))

    task.close()
