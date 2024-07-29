import json
import csv
import pdb

input_file = '../data/batch_7epBZY34pTV6xtJyHjuo5guq_output.jsonl'
output_file = '../data/openai_response.csv'


with open(output_file, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['question', 'answer'])

    openai_json_error = 0

    with open(input_file, 'r') as jsonl_file:
        for response_id, line in enumerate(jsonl_file):
            data = json.loads(line)
            content = data["response"]["body"]["choices"][0]["message"]["content"]

            try:
                questions_and_answers = json.loads(content)["questions_and_answers"]
            except json.JSONDecodeError:
                openai_json_error += 1
                if openai_json_error % 100 == 0:
                    print(f"JSON decode error at response {response_id}")
                continue

            if questions_and_answers:
                # pdb.set_trace()
                for qa in questions_and_answers:
                    if isinstance(qa, dict) and "question" in qa and "answer" in qa:
                        question = qa["question"]
                        answer = qa["answer"]
                        csv_writer.writerow([question, answer])
                    else:
                        print(f"Invalid QA format at response {response_id}")
            else:
                print(f"No questions_and_answers at response {response_id}")
