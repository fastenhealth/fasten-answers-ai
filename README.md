# Fasten Answers AI

This project uses Llama3 8B powered by llama.cpp to provide LLM answers to users' questions using documents indexed in Elasticsearch if available. The architecture includes a data flow that allows document uploads, indexing in Elasticsearch, and response generation to queries using a large language model (LLM).


## Project Summary

It uses the following components:

- **Elasticsearch**: For document storage and search.
- **llama.cpp**: For response generation using llama3 8B.
- **FastAPI**: To provide the web API.

## Data Flow Architecture

1. **Document Upload**:
   - PDF documents are uploaded via a FastAPI endpoint.
   - The document text is extracted and split into chunks using `langchain`.

2. **Indexing in Elasticsearch**:
   - The text chunks are indexed in Elasticsearch.
   - Text embeddings are generated using `sentence-transformers` and stored in Elasticsearch to facilitate search.

3. **Response Generation**:
   - Queries are sent through a FastAPI endpoint.
   - Relevant results are retrieved from Elasticsearch.
   - An LLM, served by llama.cpp, generates a response based on the retrieved results.

## Running the Project

### Prerequisites

- Docker
- Docker Compose

### Instructions

1. **Clone the repository**:

    ```sh
    git clone <https://github.com/fastenhealth/fasten-answers-ai.git>
    cd fasten-answers-ai
    ```

2. **Modify the `docker-compose.yml` file variables** (if necessary):

    ```sh
    ES_HOST=http://elasticsearch:9200
    ES_USER=elastic
    ES_PASSWORD=changeme
    EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
    LLAMA_HOST=http://llama:8080
    LLAMA_PROMPT="A chat between a curious user and an intelligent, polite medical assistant. The assistant provides detailed, helpful answers to the user's medical questions, including accurate references where applicable."
    INDEX_NAME=fasten-index
    UPLOAD_DIR=/app/data/
    ```

3. **Start the services with Docker Compose**:

    ```sh
    docker-compose up --build
    ```

    This command will start the following services:
    - **Elasticsearch**: Available at `http://localhost:9200`
    - **Llama3**: Served by llama.cpp at `http://localhost:8080`
    - **FastAPI Application**: Available at `http://localhost:8000`

## Generating CURL Requests

### Generate CURL Requests with `curl_generator`

To facilitate generating CURL requests to the 8000 port, you can use the `curl_generator.py` script located in the scripts folder.
