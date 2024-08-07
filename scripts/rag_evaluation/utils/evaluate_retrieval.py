import json
import requests

from clearml import Task

# Load your JSON data
with open('json_sample_parsed.json', 'r') as f:
    parsed_data = json.load(f)

entry_dict = {str(entry["node_id"]): entry for entry in parsed_data["entry"]}

openai_responses = []
with open('openai_responses.jsonl', 'r') as f:
    for line in f:
        openai_responses.append(json.loads(line))

# Define the endpoint URL
search_url = "http://0.0.0.0:8000/search"

# Initialize counters
total_questions = 0
total_contexts_found = 0
position_sum = 0
k = 5

# Initialize clearml task
task = Task.init(project_name="Fasten", task_name="Retrieval evaluation")


# Iterate over the OpenAI responses
for response in openai_responses:
    custom_id = response["custom_id"]
    questions_and_answers = json.loads(
        response["response"]["body"]["choices"][0]["message"]["content"]
    )["questions_and_answers"]

    context = entry_dict.get(custom_id)

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
            response = requests.get(search_url, params=params)
            search_results = response.json()

            # Check if the context is in the results and record its position
            context_found = False
            for i, result in enumerate(search_results):
                if context_str in result["content"]:
                    total_contexts_found += 1
                    position_sum += i + 1  # Found at position i+1 (1-indexed)
                    context_found = True
                    break

            if not context_found:
                position_sum += k + 1  # Assign k+1 for not found contexts

# Calculate metrics
retrieval_accuracy = total_contexts_found / total_questions if total_questions > 0 else 0
average_position = position_sum / total_questions if total_questions > 0 else 0

print(f"Retrieval Accuracy: {retrieval_accuracy:.2f}")
print(f"Average Position of Found Contexts: {average_position:.2f}")
