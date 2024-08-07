import argparse
import json
import random
import requests

from clearml import Task


def calculate_retrieval_metrics(num_sampled_questions,
                                endpoint_url):
    # Iterate over the OpenAI responses
    for response in openai_responses:
        custom_id = response["custom_id"]
        questions_and_answers = json.loads(
            response["response"]["body"]["choices"][0]["message"]["content"]
        )["questions_and_answers"]

        context = entry_dict.get(custom_id)
        
        if num_sampled_questions and len(questions_and_answers) > 0:
            questions_and_answers = random.choice(questions_and_answers)

        if context:
            context_str = context["resource"]

            for qa in questions_and_answers:
                question = qa["question"]
                total_questions += 1

                # Query the search endpoint
                params = {
                    'query': question,
                    'k': k 
                }
                response = requests.get(endpoint_url, params=params)
                search_results = response.json()

                for i, result in enumerate(search_results):
                    if context_str in result["content"]:
                        total_contexts_found += 1
                        position_sum += i + 1  # Found at position i+1 (1-indexed)
                        break
                    
    retrieval_accuracy = total_contexts_found / total_questions if total_questions > 0 else 0
    average_position = position_sum / total_questions if total_questions > 0 else 0
                    
    return retrieval_accuracy, average_position


if __name__ == "__main__":
    # Define input parameters
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics")

    parser.add_argument('experiment_iteration',
                        type=int,
                        help='The iteration number of the current experiment in a series.')
    parser.add_argument('num_sampled_questions',
                        type=int,
                        help='The number of questions to sample from the set of questions generated by OpenAI.')
    parser.add_argument('endpoint_url',
                        type=str,
                        default="http://0.0.0.0:8000/search",
                        help='The URL with the endpoint to query for search results.')

    args = parser.parse_args()
    
    experiment_iteration = args.experiment_iteration
    num_sampled_questions = args.num_sampled_questions
    endpoint_url = args.endpoint_url
    
    # Initialize clearml task and endpoint URL
    task = Task.init(project_name="Fasten",
                     task_name=f"Retrieval evaluation {experiment_iteration}")
    
    # Load JSON data
    with open('../data/json_sample_parsed.json', 'r') as f:
        parsed_data = json.load(f)

    entry_dict = {str(entry["node_id"]): entry for entry in parsed_data["entry"]}

    openai_responses = []
    with open('../data/batch_3haP1i8Dsfohov5umgZanshX_output.jsonl', 'r') as f:
        for line in f:
            openai_responses.append(json.loads(line))

    # Initialize counters
    total_contexts = len(entry_dict)
    total_questions = 0
    total_contexts_found = 0
    position_sum = 0
    k = 5

    # Evaluate metrics
    retrieval_accuracy, average_position = calculate_retrieval_metrics(num_sampled_questions, 
                                                                       endpoint_url)
    
    metrics = {
        "Retrieval Accuracy": retrieval_accuracy,
        "Average Position": average_position,
        "Total contexts found": total_contexts_found,
        "Total questions": total_questions,
        "Total contexts": total_contexts
    }
    
    for series_name, value in metrics.items():
        task.get_logger().report_scalar(
            title="Retrieval Evaluation metric",
            series=series_name,
            value=value,
            iteration=0
        )
    
    task.close()

