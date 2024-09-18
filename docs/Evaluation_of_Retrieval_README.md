# Evaluation of Retrieval for fasten-answers-ai RAG

This document provides step-by-step instructions on how to evaluate the retrieval component of the system. Follow the instructions below carefully to ensure correct execution and evaluation.

## 1. Bulk Load Data into Elasticsearch

Before starting the evaluation, the first step is to load the data into Elasticsearch using the `/database/bulkload` endpoint.

### Endpoint:
```python
@router.post("/bulk_load")
async def bulk_load(file: UploadFile = File(...), text_key: str = Form(...)):
```

- **file**: JSON or CSV containing the data.
- **text_key**: Specifies the text field for embeddings.

#### Supported Formats:
- **JSON**: Uploads FHIR resources as individual chunks into the database.
- **CSV**: Vectorizes the specified `text_key` column using sentence transformers (`all-MiniLM-L6-v2`).

Metadata such as `resource_id`, `resource_type`, `tokens_evaluated`, `tokens_predicted`, `prompt_ms`, and `predicted_ms` are saved into Elasticsearch under each document's metadata.

## 2. Generate Summaries and Load Data in Parallel

To generate summaries for each resource, use the `/generation/summarize_and_load_parallel` endpoint. This method runs summaries in parallel using LLaMA or Phi models.

### Endpoint:
```python
@router.post("/summarize_and_load_parallel")
async def summarize_and_load(file: UploadFile, remove_urls: bool, batch_size: int, limit: int):
    # Summarize resources in parallel and store results
```

#### Parameters:
- **file**: FHIR data in JSON format.
- **remove_urls**: Option to remove URLs from the original FHIR file.
- **batch_size**: Number of resources to summarize in parallel (recommended to use `1` due to strange behavior with higher values).
- **limit**: Limit the number of summaries to generate (useful for testing).

Generated files are stored in the `./data/` directory.

## 3. Generate Summaries Using OpenAI API

Alternatively, summaries can be generated using OpenAI’s API via the `/openai/execute_batch_chat_requests` endpoint.

### Endpoint:
```python
@router.post("/execute_batch_chat_requests")
async def execute_batch_chat_requests(
    openai_api_key: str, task: str, remove_urls: bool, get_costs: bool, file: UploadFile, ...
):
    # Generate summaries using OpenAI API and calculate costs
```

#### Parameters:
- **openai_api_key**: API key for OpenAI.
- **task**: Task to execute (currently supports "summarize").
- **remove_urls**: Option to remove URLs from the FHIR file.
- **get_costs**: If `True`, only returns token costs.
- **file**: FHIR data in JSON format.

The system will use the `summaries_openai_system_prompt` to guide the summary generation. The generated file is stored in the `./data/` directory.

## 4. Evaluate the Retrieval

Once the data is loaded into Elasticsearch, and summaries are generated, use the `/evaluation/evaluate_retrieval` endpoint to evaluate the retrieval performance.

### Endpoint:
```python
@router.post("/evaluate_retrieval")
async def evaluate_retrieval(file: UploadFile, index_name: str, size: int, search_text_boost: float, ...):
    # Evaluate the retrieval results using the specified parameters
```

#### Parameters:
- **file**: JSONL file containing the reference questions and answers.
- **index_name**: The Elasticsearch index to search.
- **search_text_boost**: Text field boost for search.
- **search_embedding_boost**: Embedding field boost for search.
- **k**: Number of top documents to retrieve.
- **rerank_top_k**: Rerank the top-k retrieved documents.
- **chunk_size**: Specify the chunk size if documents were divided into chunks.
- **clearml_track_experiment**: Whether to track the experiment in ClearML.

The evaluation returns metrics based on the retrieval performance, which are also uploaded to ClearML for tracking.

## 5. Final Steps

Once you have the questions and answers file for each resource, you can evaluate the retrieval performance using the above endpoint. Make sure to tune the `search_text_boost` and `search_embedding_boost` parameters. In our tests, we achieved optimal results with values of `0.25` and `4.0`, respectively.

---

By following these steps, you will be able to successfully evaluate the retrieval system and track the performance using ClearML.
