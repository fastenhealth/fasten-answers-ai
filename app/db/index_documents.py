def bulk_load_from_json_flattened_file(data: dict,
                                       embedding_model,
                                       index_name):
    """
    Function to load in bulk mode a FHIR JSON flattened file
    """
    data = data["entry"]
    for value in data:
        resource_id = value.get("resource_id")
        text_chunk = value.get("resource")
        embedding = embedding_model.encode(text_chunk)
        yield {
            "_index": index_name,
            "_source": {
                "content": text_chunk,
                "embedding": embedding,
                "metadata": {
                    "resource_id": resource_id
                }
            }
        }
