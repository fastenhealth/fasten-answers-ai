import requests


def query_rag_server(server_url: str,
                     query: str,
                     k: int,
                     threshold: float,
                     stream: bool,
                     text_boost: float,
                     embedding_boost: float):
    """
    Send a request to the RAG server with the specified query.
    """
    params = {
        "query": query,
        "k": k,
        "threshold": threshold,
        "stream": stream,
        "text_boost": text_boost,
        "embedding_boost": embedding_boost
    }
    try:
        response = requests.get(server_url, params=params)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
