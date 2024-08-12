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

QUESTION_GEN_SYS_TMPL = (
    "You are a healthcare expert specializing in patient communication. Your task is to create"
    " up to {num_questions_per_chunk} clear and contextually accurate questions and answers"
    " based on the provided flattened FHIR data. These questions will be used to evaluate"
    " the retrieval performance of a Retrieval-Augmented Generation (RAG) system. Effective"
    " questions should help retrieve the relevant data chunk from the vector database when"
    " queried."
    " Assess the coherence and meaningfulness of the context. If the context lacks sufficient"
    " information or clarity to form meaningful questions and answers, do not generate them."
    " Focus on questions that clarify specific details relevant to the context and avoid"
    " making inferences beyond the provided data. Avoid asking about dates or classifications"
    " unless directly relevant. Each answer should be thorough and directly address the patient's"
    " question."
)

QUESTION_GEN_USER_TMPL = (
    "Context information is provided below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Based solely on the provided context information, without relying on prior knowledge, "
    "generate a JSON object containing keys and values for relevant questions and answers. "
    "Ensure that your response strictly follows this format: "
    "{{\"questions_and_answers\": ["
    "{{\"question\": \"example question\", \"answer\": \"example answer\"}}"
    "]}}."
    " If the context does not provide enough information to generate any questions and answers, "
    "do not generate any output."
)
