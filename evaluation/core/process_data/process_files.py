import json
import pandas as pd


def jsonl_dataset_to_dataframe(jsonl_file, output_csv) -> pd.DataFrame:
    """
    Read evaluation dataset in JSONL format from Openai
    """
    openai_responses = []
    with open(jsonl_file, "r") as f:
        for line in f:
            openai_responses.append(json.loads(line))

    results = []

    for response in openai_responses:
        resource_id = response["custom_id"]
        content = response["response"]["body"]["choices"][0]["message"]["content"]

        try:
            questions_and_answers = json.loads(content)["questions_and_answers"][0]
            question = questions_and_answers["question"]
            reference_answer = questions_and_answers["answer"]
        except json.JSONDecodeError as e:
            print(e)
            continue

        result = {"resource_id_source": resource_id, "openai_query": question, "openai_answer": reference_answer}
        results.append(result)

    data = pd.DataFrame(results)
    data.to_csv(output_csv, index=False)

    return data
