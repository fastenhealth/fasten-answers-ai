from dotenv import load_dotenv
import logging
import os

import tiktoken


load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MAX_TOKENS = 2000

OPENAI_MODEL = "gpt-4o-mini"

OPENAI_MODEL_EMBEDDING = tiktoken.get_encoding("o200k_base")

QUESTION_GEN_SYS_TMPL = """
You are a knowledgeable and professional teacher/professor. \
Your task is to set up {num_questions_per_chunk} questions with \
their respective answers for an upcoming quiz/examination. \
Ensure the questions are diverse, relevant, and strictly adhere \
to the provided context. Avoid any sensitive or inappropriate \
topics, and focus on generating clear, informative, and contextually \
accurate questions. Do not include questions about codes, coding \
systems, or identifiers. Be concise in your answers.
"""

QUESTION_GEN_USER_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge,"
    "just generate a json of keys and values of the relevant"
    "questions and answers, and do not add more information"
)
