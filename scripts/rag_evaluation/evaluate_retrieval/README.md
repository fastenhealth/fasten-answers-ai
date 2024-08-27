
# Retrieval Metrics for FHIR-based RAG System

This repository contains two methodologies for calculating retrieval metrics on a Retrieval-Augmented Generation (RAG) system using FHIR data. These methodologies focus on evaluating the retrieval accuracy, mean reciprocal rank (MRR), average precision, and recall for different types of input data. 

Each methodology is based on a different approach to structuring and querying the data during the retrieval process.

## Methodology 1: Chunk-Level Retrieval Evaluation

### Overview
In this methodology, the system generates a question and answer for each individual chunk of a resource. The retrieval metrics are calculated based on how well the system can retrieve relevant chunks of data that contain the specific context from which the question was generated.

### Steps:
1. **Data Structure**: The FHIR data is split into chunks. Each chunk corresponds to a part of the `resource_id`.
2. **Question Generation**: For each chunk, a question and answer are generated using OpenAI's API.
3. **Retrieval**: The system retrieves a list of chunks in response to the question.
4. **Metrics Calculation**:
   - **Retrieval Accuracy**: Measures if the correct chunk containing the context is present in the retrieved results.
   - **Average Position**: The rank position of the first correct chunk in the list of retrieved results.
   - **MRR (Mean Reciprocal Rank)**: The reciprocal of the rank of the first relevant chunk, averaged over all questions.
   - **Precision**: Proportion of relevant chunks out of all chunks retrieved.
   - **Recall**: Since only one relevant chunk exists per query, recall is either 1 or 0 depending on whether the correct chunk is retrieved.

### Code Reference:
The implementation for this methodology can be found in the function `methodlogy_1_retrieval_metrics`.

---

## Methodology 2: Resource-Level Retrieval Evaluation

### Overview
In this second methodology, instead of generating a question and answer per chunk, a question and answer are generated for each complete `resource_id`. This allows for more context to be included in the question and answer, and the retrieval metrics are evaluated based on whether the returned chunks belong to the same resource from which the question was generated.

### Steps:
1. **Data Structure**: Each `resource_id` is split into chunks, but the retrieval metrics are calculated at the `resource_id` level.
2. **Question Generation**: A question and answer are generated for each `resource_id`, rather than for individual chunks.
3. **Retrieval**: The system retrieves a list of chunks in response to the question.
4. **Metrics Calculation**:
   - **Retrieval Accuracy**: Measures if any chunk belonging to the correct `resource_id` is present in the retrieved results.
   - **Average Position**: The rank position of the first correct chunk (from the `resource_id`) in the list of retrieved results.
   - **MRR (Mean Reciprocal Rank)**: The reciprocal of the rank of the first relevant chunk from the `resource_id`, averaged over all questions.
   - **Precision**: Proportion of relevant chunks (chunks that belong to the correct `resource_id`) out of all chunks retrieved.
   - **Recall**: The proportion of relevant chunks (from the correct `resource_id`) retrieved out of the total number of relevant chunks for that resource.

### Code Reference:
The implementation for this methodology can be found in the function `methodlogy_2_retrieval_metrics`.

---

## Use Cases
- **Methodology 1**: This approach is useful when dealing with very fine-grained data and when precise chunk-level context retrieval is required. It is particularly effective when the chunks are small and distinct enough that generating questions for each chunk provides valuable insight.
  
- **Methodology 2**: This approach is more appropriate when each `resource_id` contains more significant context and a holistic retrieval is desired. The larger context helps generate more robust questions, and the evaluation is based on retrieving the correct chunks associated with the resource rather than individual small chunks.

## Running the Code
Both methodologies follow a similar setup process. To run the retrieval metrics calculation:

1. **Install dependencies**: Make sure you have installed any necessary dependencies such as `requests`, `argparse`, etc.
2. **Configure your data**: Make sure you have prepared your FHIR data as a dictionary (`entry_dict`) and have the corresponding OpenAI responses (`openai_responses`).
3. **Run the function**: Call either `methodlogy_1_retrieval_metrics` or `methodlogy_2_retrieval_metrics`, passing in the required parameters.

Example command:

```python
metrics = methodlogy_1_retrieval_metrics(entry_dict, openai_responses, num_sampled_questions, endpoint_url)
```

## Conclusion
These two methodologies provide different perspectives on retrieval evaluation, offering flexibility depending on the structure of your data and the granularity of the retrieval task. Choose the methodology that best fits your needs for evaluating a FHIR-based RAG system.
