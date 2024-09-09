import json
import random

from tqdm import tqdm

from app.services.search_documents import search_query


def evaluate_resources_summaries_retrieval(
    es_client: str,
    embedding_model: str,
    resource_chunk_counts: dict,
    qa_references: list[dict],
    search_text_boost: int = 1,
    search_embedding_boost: int = 1,
    k: int = 5
) -> dict:
    # Initialize counters and sums for metrics
    total_questions = 0
    total_contexts_found = 0
    position_sum = 0
    reciprocal_rank_sum = 0
    precision_sum = 0
    recall_sum = 0

    # Iterate over the OpenAI responses
    for response in tqdm(qa_references, total=len(qa_references), desc="Calculating retrieval metrics"):
        # Get content and id of openai responses
        reference_resource_id = response["custom_id"]
        content = response["response"]["body"]["choices"][0]["message"]["content"]

        questions_and_answers = json.loads(
            content)["questions_and_answers"]

        if len(questions_and_answers) > 0:
            # Sample one random question per resource_id to evaluate
            questions_and_answers = [random.choice(questions_and_answers)]

            for qa in questions_and_answers:
                if isinstance(qa, dict) and "question" in qa:
                    question = qa["question"]
                    total_questions += 1

                    # Query question
                    search_results = search_query(question,
                                                  embedding_model,
                                                  es_client,
                                                  k=k,
                                                  text_boost=search_text_boost,
                                                  embedding_boost=search_embedding_boost)

                    # Evaluate if any returned chunk belongs to the correct resource_id
                    found = False
                    rank = 0
                    retrieved_relevant_chunks = 0

                    # Get the total number of relevant chunks for this resource_id
                    relevant_chunks = resource_chunk_counts[reference_resource_id]

                    if search_results != {"detail": "Not Found"}:
                        for i, result in enumerate(search_results):
                            if result["metadata"]["resource_id"] == reference_resource_id:
                                if not found:
                                    total_contexts_found += 1
                                    rank = i + 1
                                    reciprocal_rank_sum += 1 / rank
                                    found = True
                                retrieved_relevant_chunks += 1
                    elif search_results == {"detail": "Not Found"}:
                        search_results = {}

                    # Calculate precision and recall for this specific question
                    precision = retrieved_relevant_chunks / \
                        len(search_results) if len(search_results) > 0 else 0
                    recall = retrieved_relevant_chunks / relevant_chunks if relevant_chunks > 0 else 0

                    precision_sum += precision
                    recall_sum += recall

                    if found:
                        position_sum += rank

    # Calculate final metrics
    retrieval_accuracy = round(
        total_contexts_found / total_questions, 3) if total_questions > 0 else 0
    average_position = round(
        position_sum / total_contexts_found, 3) if total_contexts_found > 0 else 0
    mrr = round(reciprocal_rank_sum / total_questions,
                3) if total_questions > 0 else 0
    average_precision = round(
        precision_sum / total_questions, 3) if total_questions > 0 else 0
    average_recall = round(recall_sum / total_questions,
                           3) if total_questions > 0 else 0

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
        "Total contexts found": total_contexts_found,
        "Total positions sum": position_sum,
    }
