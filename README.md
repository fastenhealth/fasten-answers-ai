# Fasten Answers AI

This project uses llama.cpp to provide LLM answers to users' questions using documents indexed in Elasticsearch if available. The architecture includes a data flow that allows document uploads, indexing in Elasticsearch, and response generation to queries using a large language model (LLM).


## Project Summary

It uses the following components:

- **Elasticsearch**: For document storage and search.
- **llama.cpp**: For response generation using llama3 8B.
- **FastAPI**: To provide the web API.

## Data Flow Architecture

1. **Indexing in Elasticsearch**:
   - FHIR resources must be in JSON format.
   - As the project is still in development, we are testing different approaches for storing FHIR resources in the vector database. The alternatives we have tested include: i. saving each resource as a string, divided into chunks with overlap, ii. flattening each resource before chunking with overlap, and iii. summarizing each FHIR resource using the OpenAI API or a local LLM with llama.cpp. You can find more details on each approach in the [indexing strategies documentation](./docs/indexing_strategies.md)
   - Text embeddings are generated using `sentence-transformers` and stored in Elasticsearch.

2. **Response Generation**:
   - Queries are sent through a FastAPI endpoint. 
   - Relevant results are retrieved from Elasticsearch.
   - An LLM, served by llama.cpp, generates a response based on the retrieved results.
   - You can find more details on how to setup the generation in the [generation strategies documentation](./docs/indexing_strategies.md).

## Hardware and technical recommendations

Since the project utilizes llama.cpp for LLM execution, it’s crucial to configure the command properly to optimize performance based on the available hardware resources. All parameters can be found in the [llama.cpp server documentation](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md), but for this project, we use the following configuration:

| Parameter    | Explanation |
| -------- | ------- |
| -n, --predict, --n-predict N | number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled) |
| -c, --ctx-size N | size of the prompt context as a power of 2 (default: 0, 0 = loaded from model)     |
| -t, --threads N | number of threads to use during generation (default: -1). The higher, the faster the inference will be. |
| -np, --parallel N (optional)| number of parallel sequences to decode (default: 1). If for example you decide to process 4 sequences in a batch in parallel, you shoud set -np 4. |

To configure these parameters based on the available resources of the local machine, modify the `command` line in the `llama` service section within the [docker-compose.yml](docker-compose.yml) file directly.

If the hardware has sufficient resources to run llama.cpp in parallel, it is recommended to include the -np parameter in the Docker Compose configuration before running `docker compose up`. In this scenario, the endpoints that support parallel execution, as specified in [routes](./app/routes/), are documented in [evaluate_generation.md](./docs/evaluate_generation.md) and [evaluate_retrieval.md](./docs/evaluate_retrieval.md).

## Running the Project

### Prerequisites

- Docker.
- Docker Compose.
- LLM models should be downloaded and stored in the [./models](./models/) folder in .gguf format. We have tested the performance of Phi 3.5 Mini and Llama 3.1 in various quantization formats. The prompts for conversation and summary generation are configured in the [prompts folder](./app/config/prompts/). If you want to add a new model with a different prompt, you must update the prompt files in that directory and place the corresponding model in the models folder.

### Instructions to Launch the RAG System

1. **Clone the repository**:

    ```sh
    git clone <https://github.com/fastenhealth/fasten-answers-ai.git>
    cd fasten-answers-ai
    ```

2. **Modify the `docker-compose.yml` file app env variables** (if necessary):

    ```sh
    ES_HOST=http://elasticsearch:9200
    ES_USER=elastic
    ES_PASSWORD=changeme
    ES_INDEX_NAME: fasten-index
    EMBEDDING_MODEL_NAME: all-MiniLM-L6-v2
    LLM_HOST: http://llama:9090
    ```

3. **Start the services with Docker Compose**:

    ```sh
    docker-compose up --build
    ```

    This command will start the following services:
    - **Elasticsearch**: Available at `http://localhost:9200`
    - **Llama**: Served by llama.cpp at `http://localhost:8080`
    - **FastAPI Application**: Available at `http://localhost:8000`

## Running the evaluations

* [Retrieval evaluation](./docs/evaluate_retrieval.md)
* [Generation evaluation](./docs/evaluate_generation.md)