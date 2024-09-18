Para hacer la evaluación del retrieval se deben seguir los siguientes pasos

Se debe en primer lugar hacer la carga a la base de datos de elastic search usando el endpoint /database/bulkload

Este endpoint acepta un file y un text key.

Este es el endpoint:

@router.post("/bulk_load")
async def bulk_load(file: UploadFile = File(...), text_key: str = Form(...)):
    data = await file.read()

    if file.filename.endswith(".json"):
        json_data = json.loads(data)["entry"]
    elif file.filename.endswith(".csv"):
        json_data = csv_to_dict(data)

    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Only JSON and CSV are supported.")

    try:
        helpers.bulk(
            es_client,
            bulk_load_fhir_data(
                json_data, text_key, embedding_model=embedding_model, index_name=settings.elasticsearch.index_name
            ),
        )
        logger.info(f"Bulk load completed for file: {file.filename}")
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        logger.error(f"Bulk load failed: {str(e)}")
        return {"status": "error", "message": str(e)}


Si el inputfile en un json FHIR file, el endpoint cargará cada resource por separado como un chunk completo a la base de datos. Si es un csv y se especifica la columna de texto que se quiere vectorizar con el text_key, entonces se hará el embedding de la columna texto usando sentence transformers con el modelo de all-MiniLM-L6-v2.

El archivo debe contener los campos de resource_id y resource_type. Son opcionales los campos de tokens_evaluated,
tokens_predicted, prompt_ms, y predicted_ms. TOdos estos desde resource id hasta predicted_ms se guardarán en el campo de metadata de cada documento en elasticsearch

Para generar un archivo que se pueda subir a la base de datos, se puede usar el endpoint de generation/summarize_and_load_parallel, el cual ejecutará de forma paralela llama.cpp para generar resúmenes de cada resource usando el prompt que se especifica en en self.summaries_model_prompt en ModelsSettings, en el archivo config/settings. Se puede seleccionar entre el prompt de llama3 y el de phi3.5 mini. En una siguiente versión se colocará este parámetro como parámetro de entrada del endpoint para mayor facilidad.

Como otros parámetros de entrada, este endpoint acepta remover urls o no dentro del archivo fhir original, un batch_size que permite especificar cuantos resource resumir en paralelo. Por el momento se recomienda ejecutar un batch size de 1, ya que se nota un comportamiento extraño en la generación usando batch superiores a 1. Finalmente, limit se encarga de limitar la cantidad de resúmenes a generar, en caso de que se quiera hacer una prueba previa.

Este es el endpoint de summarize_and_load_parallel:

@router.post("/summarize_and_load_parallel")
async def summarize_and_load(
    file: UploadFile = File(...),
    remove_urls: bool = Form(True),
    batch_size: int = Form(4),
    limit: int = Form(None),
):
    # Read file
    try:
        resources = await file.read()
        resources = json.loads(resources)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON format.")
    # Process file and summarize resources
    try:
        limit = 1 if limit <= 0 else limit
        if limit:
            resources_processed = process_resources(
                data=resources, remove_urls=remove_urls)[:limit]
        else:
            resources_processed = process_resources(
                data=resources, remove_urls=remove_urls)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error during processing: {str(e)}")
    # Generate summaries and save
    try:
        output_file = await summarize_resources_parallel(model_prompt=settings.model.summaries_model_prompt,
                                                         es_client=es_client,
                                                         embedding_model=embedding_model,
                                                         resources=resources_processed,
                                                         batch_size=batch_size)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error during summaries generation: {str(e)}")

    return {"detail": "Data summarized and loaded successfully.", "Output file": output_file}


Los archivos que se generen usando este endpoint se almacenarán en el volumen ./data/ en la raiz del repositorio.

Una vez se tenga el archivo generado, se puede usar el endpoint de bulk load para guardar el archivo en la base de elastic.

Otra forma de generar un archivo con resúmenes, es usando la la API de Openai y el endpoint de openai/execute_batch_chat_requests. 

Este es el endpoint:

@router.post("/execute_batch_chat_requests")
async def execute_batch_chat_requests(
    openai_api_key: str = Form(...),
    task: str = Form("summarize"),
    remove_urls: bool = Form(True),
    get_costs: bool = Form(True),
    cost_per_million_input_tokens: float = Form(0.150),
    cost_per_million_output_tokens: float = Form(0.600),
    max_tokens_per_response: int = Form(300),
    openai_model: str = Form("gpt-4o-mini-2024-07-18"),
    file: UploadFile = File(...),
):
    # Read file
    try:
        resources = await file.read()
        resources = json.loads(resources)
    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON format.")

    # Process resources
    try:
        resources_processed = process_resources(data=resources, remove_urls=remove_urls)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error during processing: {str(e)}")

    # Get model prompt for the task
    if task == "summarize":
        system_prompt = settings.model.summaries_openai_system_prompt
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported task.")

    # Calculate costs
    if get_costs:
        try:
            costs = calculate_costs(
                system_prompt=system_prompt,
                user_prompts=resources_processed,
                cost_per_million_input_tokens=cost_per_million_input_tokens,
                cost_per_million_output_tokens=cost_per_million_output_tokens,
                tokens_per_response=max_tokens_per_response,
                model=openai_model,
            )
            return costs
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error calculating costs: {str(e)}")
    # Get answers from resources
    else:
        try:
            output_file = process_prompts_and_save_responses(
                task=task,
                system_prompt=system_prompt,
                user_prompts=resources_processed,
                openai_api_key=openai_api_key,
                model=openai_model,
                max_tokens=max_tokens_per_response,
            )
            return {"message": f"Responses saved to {output_file}"}
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error generating responses: {str(e)}")

Este endpoint recive el api key de open ai, la tarea (en este momento solo soporta la task de summarize), si se remueven urls o no al procesar los datos de fhir, el file de FHIR en formato json. Si get_costs es true, solo retorna los costos totales de  input y output tokens para la tarea que se ejecutará. Finalmente también incluye los cost_per_million_input_tokens y cost_per_million_output_tokens para que se incluyan los costos actuales al momento de ejecutar el código, y también la cantidad máxima de tokens que se pueden generar en su api y el modelo a usar. Este archivo generado también se almacena en el volumen de data en la raiz del proyecto.


Con estos dos endpoints se pueden generar la data para usar los archivos que se desean cargar con el endpoint de bulk load y emepzar a evaluar el retrieval.


Antes de proceder con el evaluation retrieval, se debe contar con el archivo que tiene las preguntas y respuestas o queries y answers para cada resource dentro del archivo FHIR para hacer la evaluación del retrieval y del generation. Decidimos hacer una pregunta y respuesta por recurso, teniendo en cuenta que usando el resource completo, podemos generar preguntas y respuestas de mayor valor. A diferencia de los frameworks de evaluación de rags como llamaindex o ragas, decidimos tomar este approach, ya que si se divide cada resource en chunks y luego se saca una pregunta y respuesta por chunk, las preguntas y respuestas generadas no son de mucho valor y terminan siendo muy similares a las de otros resources.

Actualmente el archivo con las preguntas y respuestas generadas por openai se pueden generar a partir del archivo [main.py](../evaluation/evaluation_dataset/full_json_dumps_strategy/main.py), el cual posteriormente será migrado como endpoint al archivo [openai_endpoints.py](../app/routes/openai_endpoints.py).


Una vex los datos estń cargados y se tengan las preguntas y respuestas para evaluar el retrieval, se puede usar el endpoint de evaluation/evaluate_retrieval.


Este es el endpoint de evaluate retrieval:

@router.post("/evaluate_retrieval")
async def evaluate_retrieval(
    file: UploadFile = File(...),
    index_name: str = Form(settings.elasticsearch.index_name),
    size: int = Form(2000),
    search_text_boost: float = Form(1),
    search_embedding_boost: float = Form(1),
    k: int = Form(5),
    rerank_top_k: int = Form(0),
    urls_in_resources: bool = Form(None),
    questions_with_ids_and_dates: str = Form(None),
    chunk_size: int = Form(None),
    chunk_overlap: int = Form(None),
    clearml_track_experiment: bool = Form(False),
    clearml_experiment_name: str = Form("Retrieval evaluation"),
    clearml_project_name: str = Form("Fasten"),
):
    # Read and process reference questions and answers in JSONL
    try:
        qa_references = []

        file_data = await file.read()

        for line in file_data.decode("utf-8").splitlines():
            qa_references.append(json.loads(line))
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON format.")
    # Count total chunks by resource in database
    try:
        documents = fetch_all_documents(
            index_name=index_name, es_client=es_client, size=size)
        id, counts = np.unique([resource["metadata"]["resource_id"]
                               for resource in documents], return_counts=True)
        resources_counts = dict(zip(id, counts))
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error retrieving documents: {str(e)}")
    # Evaluate retrieval
    try:
        if clearml_track_experiment:
            params = {
                "search_text_boost": search_text_boost,
                "search_embedding_boost": search_embedding_boost,
                "k": k,
                "rerank_top_k": rerank_top_k,
                "urls_in_resources": urls_in_resources,
                "questions_with_ids_and_dates": questions_with_ids_and_dates,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
            unique_task_name = f"{clearml_experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            task = Task.init(project_name=clearml_project_name,
                             task_name=unique_task_name)
            task.connect(params)

        retrieval_metrics = evaluate_resources_summaries_retrieval(
            es_client=es_client,
            embedding_model=embedding_model,
            resource_chunk_counts=resources_counts,
            qa_references=qa_references,
            search_text_boost=search_text_boost,
            search_embedding_boost=search_embedding_boost,
            k=k,
            rerank_top_k=rerank_top_k,
        )

        # Upload metrics and close task
        if task:
            for series_name, value in retrieval_metrics.items():
                task.get_logger().report_single_value(name=series_name, value=value)

            task.close()

        return retrieval_metrics

    except Exception as e:
        logger.error(f"Error during retrieval evaluation: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error during retrieval evaluation: {str(e)}")

Ese endpoint tiene en cuenta que la base de datos está populada, por lo que antes de ejecutarlo, se puede probar el endpoint de database/get_all_documents para validar que todo esté cargado.


Ya teniendo estas dos cosas, el archivo y la base de datos, hay que especificar el search_text_boost
search_embedding_boost, los cuales son valores que elastic search permiten configurar para obtener mejores resultados al hacer una búsqueda. En nuestras pruebas, la combinación correcta nos dio que se alcanzaban mejores métricas para 0.25 y 4.0 respectivamente. k es la cantdad de documentos retornados por elastic. Rerank top ejecuta un reranker de los resultados obtenidos. urls_in_resources permite saber si dentro de los resourses en la base de datos se incluyeron o eliminaron las urls. questions_with_ids_and_dates permite saber si las preguntas generadas por openai tiene o no ids y fechas. Finalmente chunk_size y chunk_overlap permiten saber si se hizo alguna división de chunks al momento de guardar los docuemntos en elastic. Como en este momento el endpoint de bulk load no hace chunking, entonces se puede dejar vacío. Estos campos son más que todo para almacenar en clearml la configuración del experimento para posteriores análisis y comparaciones. Una vez se ejecute este endpoint, se tendrán los resultados las metricas del retrieval cargadas a clearml y como respuesta del endpoint.