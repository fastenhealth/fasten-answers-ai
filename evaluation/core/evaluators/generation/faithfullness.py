"""
- Faithfullness evaluator: The faithfulness evaluator evaluates whether the response is faithful to any of the retrieved contexts.
- This code is based on the Building Evaluation from Scratch example from llamaindex:
https://docs.llamaindex.ai/en/stable/examples/low_level/evaluation/#evaluating-generation
"""

import csv
import os
import pandas as pd
from evaluation.core.openai import get_chat_completion


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
        user_prompt = FAITHFULLNESS_USER_TMPL.format(
            generated_answer=generated_answer, contexts=contexts)
        system_prompt = FAITHFULLNESS_SYS_TMPL

        open_ai_response = get_chat_completion(
            self.openai_api_key, user_prompt, system_prompt, model=self.model, max_tokens=self.max_tokens)
        relevancy = 1 if open_ai_response["relevancy"] == "YES" else 0
        accuracy = 1 if open_ai_response["accuracy"] == "YES" else 0
        conciseness_and_pertinence = 1 if open_ai_response[
            "conciseness_and_pertinence"] == "YES" else 0
        reasoning = open_ai_response["reasoning"]

        return {
            "relevancy": relevancy,
            "accuracy": accuracy,
            "conciseness_and_pertinence": conciseness_and_pertinence,
            "reasoning": reasoning,
        }

    def run_batch_evaluation(self, 
                             df: pd.DataFrame,
                             output_file: str,
                             generated_answer_column: str,
                             contexts_column: str):
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
                                    'relevancy', 'accuracy', 'conciseness_and_pertinence', 'reasoning'])

            # Write header only if the file does not exist
            if not file_exists:
                writer.writeheader()

            try:
                for _, row in df.iterrows():
                    result = self.run_faithfulness_eval(
                        row[generated_answer_column], row[contexts_column])

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
        faithfulness_relevancy = results_df["relevancy"].sum() / total_questions
        faithfulness_accuracy = results_df["accuracy"].sum() / total_questions
        faithfulness_conciseness_and_pertinence = results_df["conciseness_and_pertinence"].sum() / total_questions
 
        return faithfulness_relevancy, faithfulness_accuracy, faithfulness_conciseness_and_pertinence
