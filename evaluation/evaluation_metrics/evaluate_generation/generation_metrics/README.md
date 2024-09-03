
# Generation Evaluation Metrics

This project is designed to evaluate the generation of responses using language models (LLMs) in a Retrieval-Augmented Generation (RAG) setup. This README provides the necessary steps to set up and run the project, including executing several scripts that perform specific tasks.

## Prerequisites

### 1. Create a Virtual Environment
Before starting, you need to create two virtual environments with Python 3.9 and install the required packages:

```bash
python3.9 -m venv venv-dev
source venv-dev/bin/activate
pip install -r requirements-dev.txt

python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Docker Configuration
This project requires setting up Elasticsearch and a Llama.cpp server. Ensure Docker is installed and run the following commands to start the necessary containers:

#### Start Elasticsearch
```bash
docker run --rm -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "xpack.security.http.ssl.enabled=false" \
  --memory="512m" \
  -v data:/usr/share/elasticsearch/data \
  --name elasticsearch \
  docker.elastic.co/elasticsearch/elasticsearch:8.12.1
```

#### Start LLama.cpp sercer
```bash
docker run --rm -p 8090:9090 -v $(pwd)/models:/models ghcr.io/ggerganov/llama.cpp:server -m models/Phi-3.5-mini-instruct-F16-Q8_0.gguf -t 11 -n 200 -c 2048 --host 0.0.0.0 --port 9090
```

Ensure that the Phi model is located in the `models/` folder and modify the parameters as needed. Change the LLM that you want to use in the docker run configuration.

### 3. Start the RAG Server
The RAG server must be active before running the generation scripts. Use the following command to start the server inside the venv environment:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Running the Scripts

### 1. Response Generation - `generate_responses.py`

This script converts an OpenAI JSONL file into a DataFrame, makes batch requests to the RAG server, and saves the responses to a CSV file. The JSONL file must be located in the ./evaluation/data/openai_outputs folder

#### Command to Run:

```bash
python -m evaluation.evaluation_metrics.evaluate_generation.generation_metrics.generate_responses
```

#### Configurable Parameters:

The parameters required for this script are found in the `data/input_generate_responses.json` file:

```json
{
    "SERVER_URL": "http://localhost:8000/generate",
    "JSONL_FILE": "batch_cn1d3YOzng9mkfZawyqfkL1k_output_33a6_ids_dates_no_urls.jsonl",
    "RAG_OUTPUT_CSV": "responses_evaluate_generation.csv",
    "CORES": 11,
    "CONTEXT_SIZE": 2048,
    "TEXT_BOOST": 0.25,
    "EMBEDDING_BOOST": 4.0
}
```

### 2. Cost Estimation - `estimate_costs.py`

This script calculates the estimated cost of using the OpenAI API to evaluate the generated responses.

#### Command to Run:

```bash
python -m evaluation.evaluation_metrics.estimate_costs
```

#### Configurable Parameters:

The parameters required for this script are found in the `data/input_estimate_costs.json` file:

```json
{
    "REFERENCE_ANSWERS_FILE": "batch_cn1d3YOzng9mkfZawyqfkL1k_output_33a6_ids_dates_no_urls.csv",
    "GENERATED_ANSWERS_FILE": "Phi3.5-responses_evaluate_generation.csv",
    "COST_PER_MILLION_INPUT_TOKENS": 0.150,
    "COST_PER_MILLION_OUTPUT_TOKENS": 0.600,
    "QUERY_COLUMN": "query",
    "CONTEXTS_COLUMN": "concatenated_contexts",
    "GENERATED_ANSWER_COLUMN": "response",
    "REFERENCE_ANSWER_COLUMN": "openai_answer"
}
```

### 3. Metrics Collection - `get_metrics.py`

This script evaluates the correctness and faithfulness of the generated responses, saves the results to CSV files, and optionally uploads artifacts to ClearML.


#### Configurable Parameters:

The parameters required for this script are found in the `data/input_get_metrics.json` file:

```json
{
    "REFERENCE_ANSWERS_FILE": "batch_cn1d3YOzng9mkfZawyqfkL1k_output_33a6_ids_dates_no_urls.csv",
    "GENERATED_ANSWERS_FILE": "Phi3.5-responses_evaluate_generation.csv",
    "CORRECTNESS_RESULTS_CSV": "correctness_results.csv",
    "FAITHFULNESS_RESULTS_CSV": "faithfulness_results.csv",
    "QUERY_COLUMN": "query",
    "CONTEXTS_COLUMN": "concatenated_contexts",
    "GENERATED_ANSWER_COLUMN": "response",
    "REFERENCE_ANSWER_COLUMN": "openai_answer",
    "RESOURCE_ID_COLUMN": "resource_id_source",
    "LLM_MODEL": "gpt-4o-mini-2024-07-18",
    "EXPERIMENT_NAME": "Generation evaluation with Phi3.5 Q8_0",
    "UPLOAD_EXPERIMENT": true,
    "UPLOAD_ARTIFACTS": true
}
```

#### Metrics Explanation:

The script evaluates the following metrics:

- **Correctness Mean Score:** This metric evaluates whether the generated answer correctly matches the reference answer provided for the query. The score ranges from 1 to 5, where a higher score indicates better correctness.

- **Faithfulness Relevancy:** This metric checks if the generated answer is relevant to the contexts provided. It evaluates whether the response focuses on and pertains to the information within the context. The accepted values are YES or NO.

- **Faithfulness Accuracy:** This metric assesses if the information provided in the generated answer is accurate and correctly reflects the context. While relevancy checks if the answer is related to the context, accuracy evaluates if the details provided are correct. The accepted values are YES or NO.

- **Faithfulness Conciseness and Pertinence:** This metric evaluates whether the generated answer avoids including unrelated or irrelevant information and remains concise. The accepted values are YES or NO.

This script also requires the `.env` file to be properly configured to access the OpenAI API with the variable name OPENAI_API_KEY.

#### Command to Run:

```bash
python -m evaluation.evaluation_metrics.get_metrics
```

