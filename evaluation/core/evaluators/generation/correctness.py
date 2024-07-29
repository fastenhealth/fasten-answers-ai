"""
- Correctness evaluator: whether the generated answer matches that of the reference answer
given the query (requires labels).
- Assuming that the answer given by the LLM is correct (ex. openai model answer with context given), 
we compare the answer of our rag vs the answer of the evaluator.
- This code is based on the Building Evaluation from Scratch example from llamaindex:
https://docs.llamaindex.ai/en/stable/examples/low_level/evaluation/#evaluating-generation
"""
import pandas as pd

from evaluation.core.openai import get_chat_completion


CORRECTNESS_SYS_TMPL = """
You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- a user query, 
- a reference answer, and
- a generated answer.

Your job is to judge the relevance and correctness of the generated answer.
Output a single score that represents a holistic evaluation.
You must return your response in a line with only the score.
Do not return answers in any other format.
On a separate line provide your reasoning for the score as well.

Follow these guidelines for scoring:
- Your score has to be between 1 and 5, where 1 is the worst and 5 is the best.
- If the generated answer is not relevant to the user query, \
you should give a score of 1.
- If the generated answer is relevant but contains mistakes, \
you should give a score between 2 and 3.
- If the generated answer is relevant and fully correct, \
you should give a score between 4 and 5.
"""

CORRECTNESS_USER_TMPL = """
## User Query
{query}

## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}
"""

ANSWER_JSON_SCHEMA = {
    "type": "json_schema",
            "json_schema": {
                "name": "correctness_evaluation_output",
                "schema": {
                    "type": "object",
                    "properties": {
                        "reasoning": {"type": "string"},
                        "score": {"type": "number"}
                    },
                    "required": ["reasoning", "score"],
                    "additionalProperties": False
                },
                "strict": True
            }
}


class CorrectnessEvaluator:
    def __init__(self, openai_api_key, model="gpt-4o-2024-08-06", threshold=4.0):
        self.openai_api_key = openai_api_key
        self.model = model
        self.threshold = threshold

    def run_correctness_eval(self, query_str: str, reference_answer: str, generated_answer: str):
        """
        Evaluates the correctness of a generated answer against a reference answer.
        
        Parameters:
        - query_str: str, the query string.
        - reference_answer: str, the reference answer.
        - generated_answer: str, the generated answer.
        
        Returns:
        - dict, containing whether the answer passes the threshold, the score, and reasoning.
        """
        user_prompt = CORRECTNESS_USER_TMPL.format(query_str,
                                                   reference_answer,
                                                   generated_answer)
        system_prompt = CORRECTNESS_SYS_TMPL

        open_ai_response = get_chat_completion(self.openai_api_key,
                                               user_prompt,
                                               system_prompt,
                                               model=self.model)
        score = open_ai_response["score"]
        reasoning = open_ai_response["reasoning"]

        return {"passing": score >= self.threshold,
                "score": score,
                "reason": reasoning}

    def run_batch_evaluation(self, df: pd.DataFrame):
        """
        Runs correctness evaluation on a batch of queries, reference answers, and generated answers.
        
        Parameters:
        - df: pd.DataFrame, a dataframe with columns 'query', 'reference_answer', and 'generated_answer'.
        
        Returns:
        - pd.DataFrame, the original dataframe with additional columns for score, reasoning, and passing status.
        """
        results = []
        for _, row in df.iterrows():
            result = self.run_correctness_eval(row['query'], row['reference_answer'], row['generated_answer'])
            results.append(result)

        # Convert list of dicts to a DataFrame
        results_df = pd.DataFrame(results)

        # Concatenate the original dataframe with the results
        df = pd.concat([df, results_df], axis=1)

        return df
