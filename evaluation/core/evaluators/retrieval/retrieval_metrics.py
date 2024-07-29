import json
import random
import requests


def methodlogy_1_retrieval_metrics(entry_dict,
                                   openai_responses,
                                   num_sampled_questions,
                                   endpoint_url,
                                   search_text_boost,
                                   search_embedding_boost,
                                   k=5):
    # Initialize counters and sums for metrics
    total_questions = 0
    total_contexts_found = 0
    position_sum = 0
    reciprocal_rank_sum = 0
    precision_sum = 0
    recall_sum = 0
    openai_json_error = 0

    # Iterate over the OpenAI responses
    for response in openai_responses:
        custom_id = response["custom_id"]

        content = response["response"]["body"]["choices"][0]["message"]["content"]

        try:
            questions_and_answers = json.loads(content)["questions_and_answers"]
        except json.JSONDecodeError as e:
            openai_json_error = +1
            if openai_json_error % 100 == 0:
                print(content)
                print(e)
            continue  # Skip to the next response if there's an error

        context = entry_dict[custom_id]

        if len(questions_and_answers) > 0:
            # Sample questions to measure
            if num_sampled_questions and len(questions_and_answers) > 0:
                questions_and_answers = [random.choice(questions_and_answers)]

            if context:
                context_str = context

                for qa in questions_and_answers:
                    if isinstance(qa, dict) and "question" in qa:

                        question = qa["question"]
                        total_questions += 1

                        # Query the search endpoint
                        params = {
                            'query': question,
                            'k': k,
                            'text_boost': search_text_boost,
                            'embedding_boost': search_embedding_boost
                        }
                        response = requests.get(endpoint_url, params=params)
                        search_results = response.json()

                        # Calculate metrics for each question
                        found = False
                        rank = 0

                        for i, result in enumerate(search_results):
                            if context_str in result["content"]:
                                if not found:
                                    total_contexts_found += 1
                                    rank = i + 1
                                    reciprocal_rank_sum += 1 / rank
                                    found = True

                                break  # Stop after finding the first relevant document

                        if found:
                            position_sum += rank
                            precision_sum += 1 / len(search_results)
                            recall_sum += 1  # Since there's only one relevant document

    # Calculate final metrics
    retrieval_accuracy = total_contexts_found / total_questions if total_questions > 0 else 0
    average_position = position_sum / total_contexts_found if total_questions > 0 else 0
    mrr = reciprocal_rank_sum / total_questions if total_questions > 0 else 0
    average_precision = precision_sum / total_questions if total_questions > 0 else 0
    average_recall = recall_sum / total_questions if total_questions > 0 else 0

    return {
        "Retrieval Accuracy": retrieval_accuracy,
        "Average Position": average_position,
        "MRR": mrr,
        "Average Precision": average_precision,
        "Average Recall": average_recall,
        "Total Questions": total_questions,
        "Total Openai json error": openai_json_error,
        "Total contexts fount": total_contexts_found,
        "Total positions sum": position_sum
    }


def methodlogy_2_retrieval_metrics(resource_chunk_counts,
                                   openai_responses,
                                   num_sampled_questions,
                                   endpoint_url,
                                   search_text_boost,
                                   search_embedding_boost,
                                   k=5):
    # Initialize counters and sums for metrics
    total_questions = 0
    total_contexts_found = 0
    position_sum = 0
    reciprocal_rank_sum = 0
    precision_sum = 0
    recall_sum = 0
    openai_json_error = 0

    # Iterate over the OpenAI responses
    for response in openai_responses:
        custom_id = response["custom_id"]

        content = response["response"]["body"]["choices"][0]["message"]["content"]

        try:
            questions_and_answers = json.loads(content)["questions_and_answers"]
        except json.JSONDecodeError as e:
            openai_json_error += 1
            if openai_json_error % 100 == 0:
                print(content)
                print(e)
            continue  # Skip to the next response if there's an error

        if len(questions_and_answers) > 0:
            # Sample one random question per resource_id to evaluate
            if num_sampled_questions and len(questions_and_answers) > 0:
                questions_and_answers = [random.choice(questions_and_answers)]

            for qa in questions_and_answers:
                if isinstance(qa, dict) and "question" in qa:
                    question = qa["question"]
                    total_questions += 1

                    # Query the search endpoint
                    params = {
                        'query': question,
                        'k': k,
                        'text_boost': search_text_boost,
                        'embedding_boost': search_embedding_boost
                    }
                    response = requests.get(endpoint_url, params=params)
                    search_results = response.json()

                    # Evaluate if any returned chunk belongs to the correct resource_id
                    found = False
                    rank = 0
                    retrieved_relevant_chunks = 0

                    # Get the total number of relevant chunks for this resource_id
                    relevant_chunks = resource_chunk_counts[custom_id]

                    if search_results != {'detail': 'Not Found'}:
                        for i, result in enumerate(search_results):
                            if result["metadata"]["resource_id"] == custom_id:
                                if not found:
                                    total_contexts_found += 1
                                    rank = i + 1
                                    reciprocal_rank_sum += 1 / rank
                                    found = True
                                retrieved_relevant_chunks += 1
                    elif search_results == {'detail': 'Not Found'}:
                        search_results = {}

                    # Calculate precision and recall for this specific question
                    precision = retrieved_relevant_chunks / len(search_results) if len(search_results) > 0 else 0
                    recall = retrieved_relevant_chunks / relevant_chunks if relevant_chunks > 0 else 0
                    precision_sum += precision
                    recall_sum += recall

                    if found:
                        position_sum += rank

    # Calculate final metrics
    retrieval_accuracy = round(total_contexts_found / total_questions, 3) if total_questions > 0 else 0
    average_position = round(position_sum / total_contexts_found, 3) if total_contexts_found > 0 else 0
    mrr = round(reciprocal_rank_sum / total_questions, 3) if total_questions > 0 else 0
    average_precision = round(precision_sum / total_questions, 3) if total_questions > 0 else 0
    average_recall = round(recall_sum / total_questions, 3) if total_questions > 0 else 0

    return {
        # The percentage of questions for which the system successfully retrieved at least one relevant chunk.
        "Retrieval Accuracy": retrieval_accuracy,
        "Average Position": average_position,
        "MRR": mrr,
        # Precision = Number of relevant chunks returned / Total number of chunks returned
        "Average Precision": average_precision,
        # Recall = Number of relevant chunks returned / Total number of relevant chunks that exist
        "Average Recall": average_recall,
        # Others
        "Total Questions": total_questions,
        "Total Openai json error": openai_json_error,
        "Total contexts found": total_contexts_found,
        "Total positions sum": position_sum
    }
