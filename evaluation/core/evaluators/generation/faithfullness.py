"""
- Faithfullness evaluator: The faithfulness evaluator evaluates whether the response is faithful to any of the retrieved contexts.
- This code is based on the Building Evaluation from Scratch example from llamaindex:
https://docs.llamaindex.ai/en/stable/examples/low_level/evaluation/#evaluating-generation
"""

import csv
import json
import logging
import os
import pandas as pd

from tqdm import tqdm

from evaluation.core.openai.openai import get_chat_completion


FAITHFULLNESS_SYS_TMPL = """
Please evaluate the faithfulness of a generated answer with respect to the given context. The context consists of chunks of data from FHIR resources, which may include structured medical codes, descriptions, and other related information.
Your evaluation should cover the following three aspects:
1. Relevancy: Does the generated answer focus only on the information contained in the context?
2. Accuracy: Is the information provided in the generated answer accurate and correctly reflects the context?
3. Conciseness and Pertinence: Does the generated answer avoid including unrelated or irrelevant information with respect to the context?

You need to answer each question with either YES or NO.
Some examples are provided below.

Information: Apple pie is generally double-crusted.
Context: An apple pie is a fruit pie in which the principal filling ingredient is apples. 
Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard or cheddar cheese.
It is generally double-crusted, with pastry both above and below the filling; the upper crust may be solid or latticed (woven of crosswise strips).
Relevancy: YES
Accuracy: YES
Conciseness and Pertinence: YES
Reasoning: The context explicitly mentions that apple pie is generally double-crusted, which supports the information.

Information: Apple pies taste bad.
Context: An apple pie is a fruit pie in which the principal filling ingredient is apples. 
Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard or cheddar cheese.
It is generally double-crusted, with pastry both above and below the filling; the upper crust may be solid or latticed (woven of crosswise strips).
Relevancy: NO
Accuracy: NO
Conciseness and Pertinence: YES
Reasoning: The context does not provide any information regarding the taste of apple pies, so the statement cannot be supported. However, the response is concise and avoids unrelated information.
"""

FAITHFULLNESS_USER_TMPL = """
## Information
{generated_answer}

## Contexts
{contexts}
"""

ANSWER_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "faithfulness_evaluation_output",
        "schema": {
            "type": "object",
            "properties": {
                "relevancy": {"type": "string"},
                "accuracy": {"type": "string"},
                "conciseness_and_pertinence": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["relevancy", "accuracy", "conciseness_and_pertinence", "reasoning"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


class FaithfulnessEvaluator:
    def __init__(self, openai_api_key, model="gpt-4o-2024-08-06", max_tokens=300):
        self.openai_api_key = openai_api_key
        self.model = model
        self.max_tokens = max_tokens

    def run_faithfulness_eval(self, generated_answer: str, contexts: str):
        """
        Evaluates the faithfulness of a generated answer against provided contexts based on three aspects.

        Parameters:
        - generated_answer: str, the generated answer.
        - contexts: str, the contexts that should support the answer.

        Returns:
        - dict, containing evaluations on relevancy, accuracy, conciseness and pertinence, and reasoning.
        """
        try:
            user_prompt = FAITHFULLNESS_USER_TMPL.format(generated_answer=generated_answer,
                                                         contexts=contexts)
            system_prompt = FAITHFULLNESS_SYS_TMPL

            open_ai_response = get_chat_completion(self.openai_api_key,
                                                   user_prompt,
                                                   system_prompt,
                                                   ANSWER_JSON_SCHEMA,
                                                   model=self.model,
                                                   max_tokens=self.max_tokens)

            json_answer = json.loads(open_ai_response.get("choices")[
                                     0].get("message").get("content"))

            relevancy = 1 if json_answer["relevancy"] == "YES" else 0
            accuracy = 1 if json_answer["accuracy"] == "YES" else 0
            conciseness_and_pertinence = 1 if json_answer[
                "conciseness_and_pertinence"] == "YES" else 0
            reasoning = json_answer["reasoning"]

            return {
                "relevancy": relevancy,
                "accuracy": accuracy,
                "conciseness_and_pertinence": conciseness_and_pertinence,
                "reasoning": reasoning,
            }

        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON response: {e}")
            return {"relevancy": None, "accuracy": None, "conciseness_and_pertinence": None, "reasoning": "Invalid JSON response"}

        except KeyError as e:
            logging.error(f"Missing key in JSON response: {e}")
            return {"relevancy": None, "accuracy": None, "conciseness_and_pertinence": None, "reasoning": "Incomplete JSON response"}

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return {"relevancy": None, "accuracy": None, "conciseness_and_pertinence": None, "reasoning": "An unexpected error occurred"}

    def run_batch_evaluation(self,
                             df: pd.DataFrame,
                             output_file: str,
                             generated_answer_column: str,
                             contexts_column: str,
                             resource_id_column: str):
        """
        Runs faithfulness evaluation on a batch of generated answers and contexts.
        Saves results incrementally to avoid data loss in case of failure.

        Parameters:
        - df: pd.DataFrame, a dataframe with columns 'generated_answer' and 'contexts'.
        - output_file: str, the path to the output CSV file where results will be saved.

        Returns:
        - pd.DataFrame, the original dataframe with additional columns for relevancy, accuracy, conciseness and pertinence, and reasoning.
        """
        # Determine if the file already exists
        file_exists = os.path.isfile(output_file)

        with open(output_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=[
                                    resource_id_column, 'relevancy', 'accuracy', 'conciseness_and_pertinence', 'reasoning'])

            # Write header only if the file does not exist
            if not file_exists:
                writer.writeheader()

            try:
                for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing faithfulness"):
                    result = self.run_faithfulness_eval(
                        row[generated_answer_column],
                        row[contexts_column])
                    result[resource_id_column] = row[resource_id_column]
                    # Write the result to the CSV file
                    writer.writerow(result)

                    # Ensure the data is written to disk
                    file.flush()

            except Exception as e:
                print(f"Error encountered: {e}. Saving progress and exiting.")
                raise

        # Load the results back into a DataFrame and concatenate with the original
        results_df = pd.read_csv(output_file)

        total_questions = len(results_df)
        faithfulness_relevancy = round(results_df["relevancy"].sum(
        ) / total_questions, 2)
        faithfulness_accuracy = round(
            results_df["accuracy"].sum() / total_questions, 2)
        faithfulness_conciseness_and_pertinence = round(results_df["conciseness_and_pertinence"].sum(
        ) / total_questions, 2)

        return faithfulness_relevancy, faithfulness_accuracy, faithfulness_conciseness_and_pertinence
