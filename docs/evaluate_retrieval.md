
# Retrieval Evaluation for fasten-answers-ai

This document provides step-by-step instructions on how to evaluate the retrieval component of the system. Follow the instructions below carefully to ensure correct execution and evaluation.

## Metrics evaluated

Here are brief explanations of the metrics we’ve used:

* **Retrieval Accuracy:** The percentage of questions for which the system successfully retrieved at least one relevant chunk.
* **Average Position:** Average position of the relevant chunks retrieved.
* **MRR (Mean Reciprocal Rank):** For each query, the rank of the first relevant document is noted, and the reciprocal of this rank (1/rank) is calculated. The average of these reciprocal ranks across all queries gives the MRR score.
* **Average Precision:** Average of the relevant chunks retrieved over the total chunks retrieved.
* **Average Recall:** Average of the relevant chunks retrieved over the total relevant chunks

## Retrieval evaluation summary

These are the necessary steps for successfully running the retrieval evaluation:

1. **Evaluation dataset:** A dataset containing the questions and answers for each resource or for the chunks of each resource, to be used for the evaluation.
2. **Populate database:** The Elasticsearch database must be populated using one of the strategies mentioned in the [indexing strategies documentation](./docs/indexing_strategies.md).
3. **Evaluate retrieval**: the retrieval evaluation by using the [/evaluation/evaluate_retrieval](../app/routes/evaluation_endpoints.py) endpoint, providing the evaluation dataset and adjusting parameters such as `search_text_boost`, `search_embedding_boost`, and `k` to fine-tune the results."

## 1. Evaluation dataset

In this case, we will use the "questions and answers per complete resource" strategy as an example. This strategy is preferred for evaluation because it allows for more comprehensive and robust questions and answers by using the entire resource context, unlike other strategies used by RAG evaluation frameworks such as LlamaIndex or RAGAS, which create datasets per chunk, resulting in repetitive and less valuable questions. Given the nature of the problem and the structure of FHIR data, we decided that generating questions per resource provides a fuller context.

Currently, the Q&A file can be generated using the file created by the [main.py script](../evaluation/evaluation_dataset/full_json_dumps_strategy/main.py), which produces a JSONL file that is compatible with the OpenAI batch API

This script will soon be migrated into an endpoint in [openai_endpoints.py](../app/routes/openai_endpoints.py)

At present, there are two JSONL files available for evaluation, both generated using the OpenAI API in batch mode. One includes questions with resource IDs and dates (if available), while the other does not. Both files were generated after removing URLs from the original resources.

* [File with IDs and dates](../evaluation/data/openai_outputs/batch_cn1d3YOzng9mkfZawyqfkL1k_output_33a6_ids_dates_no_urls.jsonl)
* [File without IDs and dates](../evaluation/data/openai_outputs/batch_O9aiHyHpLDHCaaxZzcw1dqDS_output_no_ids_dates_no_urls.jsonl)

## 2. Populate database

There are different strategies we have used to store data in Elasticsearch and improve retrieval metrics. In this case, we will use the strategy of generating a summary for each resource. This resource summary will be saved in the database. The fields `resource_id`, `resource_type`, `tokens_evaluated`, `tokens_predicted`, `prompt_ms`, and `predicted_ms` will be stored as metadata for each document. Since we are using LLaMA.cpp, the last three fields are available because they are returned in the response from LLaMA.cpp along with the requested summary.

To generate summaries for each resource and load them into the database, use the [`/generation/summarize_and_load_parallel`](../app/routes/llm_endpoints.py) endpoint. This method runs summaries in parallel using the LLaMA 3.1 or Phi-3.5 mini models.

### Summarize and load in parallel endpoint
```python
@router.post("/summarize_and_load_parallel")
async def summarize_and_load(file: UploadFile,
                             remove_urls: bool,
                             batch_size: int,
                             limit: int):
```

#### Parameters:
- **file**: FHIR data in JSON format.
- **remove_urls**: Option to remove URLs from the original FHIR file before creating the summaries.
- **batch_size**: Number of resources to summarize in parallel (it is recommended to use 1 due to unexpected behavior in the answers with higher values until the issue is fixed).
- **limit**: Limit the number of summaries to generate (useful for testing).

After generating the summaries, the resulting file can be reviewed in the path [`./data/`](../app/data/), and you can also verify that they have been successfully loaded into the database using the endpoint [`database/get_all_documents`](../app/routes/database_endpoints.py).

### Get all documents endpoint

```python
@router.get("/get_all_documents")
async def get_all_documents(index_name: str = settings.elasticsearch.index_name, size: int = 2000)
```

Finally, if you want to use an existing file to load data directly into the database without generating the summaries, you can use the file [resources_summarized.csv](../app/data/resources_summarized.csv) through the endpoint [/database/bulkload](../app/routes/database_endpoints.py).

### Bulk load data endpoint

```python
@router.post("/bulk_load")
async def bulk_load(file: UploadFile = File(...),
                    text_key: str = Form(...)):
```
#### Parameters:
- **file**: JSON or CSV containing the data.
- **text_key**: Specifies the text field to create the embeddings using sentence transformers with `all-MiniLM-L6-v2`.

#### Supported Formats:
- **JSON**: Uploads each FHIR resource as individual document into the database.
- **CSV**: Each row is uploaded as a document. The specified text_key column will contain the text that is converted into an embedding.

Metadata such as `resource_id` and `resource_type` are saved into Elasticsearch under each document's metadata. If available, the values for `tokens_evaluated`, `tokens_predicted`, `prompt_ms`, and `predicted_ms` will also be saved under each document's metadata.

Now, with the database populated and the evaluation dataset ready, you can proceed to the evaluation.

## 3. Evaluate retrieval

Once the data is loaded into Elasticsearch, and the Q&A file is ready, use the [`/evaluation/evaluate_retrieval`](../app/routes/evaluation_endpoints.py) endpoint to evaluate the retrieval performance. Make sure to tune the `search_text_boost` and `search_embedding_boost` parameters. In our tests, we achieved optimal results with values of `0.25` and `4.0`, respectively.

### Evaluate retrieval endpoint:
```python
@router.post("/evaluate_retrieval")
async def evaluate_retrieval(file: UploadFile,
                             index_name: str,
                             size: int,
                             search_text_boost: float, ...):
```

#### Parameters:
- **file**: JSONL file containing the reference questions and answers.
- **index_name**: The Elasticsearch index to search.
- **size**: The total number of documents returned by the database. In this case, we want to return all documents for evaluation and count how many chunks there are per resource. Therefore, this value is set to a large number or can be adjusted to match the total number of documents stored in the database.
- **search_text_boost**: Text field boost for search. 0.25 has been the best in our experiments.
- **search_embedding_boost**: Embedding field boost for search. 4.0 has been the best in our experiments.
- **k**: Number of top documents to retrieve
- **urls_in_resources**: Indicates whether the resources in the database or the summaries were created with or without URLs in the FHIR data.
- **questions_with_ids_and_dates**: Indicates whether the questions and answers used for the retrieval evaluation include dates and IDs or not.
- **rerank_top_k**: Rerank the top-k retrieved documents.
- **chunk_size**: Specify the chunk size if the documents were divided into chunks. This value is used for the experiment tracker.
- **chunk_overlap**: Specify the chunk overlap if documents were divided into chunks.
- **clearml_track_experiment**: Whether to track the experiment in ClearML.
- **clearml_experiment_name**: Clearml experiment name (Retrieval evaluation recommended).
- **clearml_project_name**: Clearml project name.


By following these steps, you will be able to successfully evaluate the retrieval system and track the performance using ClearML.
