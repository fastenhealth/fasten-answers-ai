def bulk_load_fhir_data(data: list[dict], text_key: str, embedding_model, index_name):
    """
    Function to load in bulk mode a FHIR data
    """
    for value in data:
        resource_id = value.get("resource_id")
        resource_type = value.get("resource_type")
        resource = value.get(text_key)
        embedding = embedding_model.encode(resource)

        metadata = {"resource_id": resource_id, "resource_type": resource_type}

        if "tokens_evaluated" in value:
            metadata["tokens_evaluated"] = value["tokens_evaluated"]
        if "tokens_predicted" in value:
            metadata["tokens_predicted"] = value["tokens_predicted"]
        if "prompt_ms" in value:
            metadata["prompt_ms"] = value["prompt_ms"]
        if "predicted_ms" in value:
            metadata["predicted_ms"] = value["predicted_ms"]

        yield {"_index": index_name, "_source": {"content": resource,
                                                 "embedding": embedding,
                                                 "metadata": metadata}}
