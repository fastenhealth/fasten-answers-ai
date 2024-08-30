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

# System prompt using date and id
QUESTION_GEN_SYS_TMPL_1 = (
    "You are a healthcare expert specializing in patient communication. Your task is to create"
    " one clear and contextually accurate question from the perspective of the patient as if you"
    " the system were the patient, based on the provided flattened FHIR data. If available, ensure"
    " the question incorporates the ID and date for referencing the specific resource. The answer"
    " should address the patient's medical inquiries about their health condition, treatments, or"
    " any medical processes documented in the FHIR data. When crafting the answer, ensure it is"
    " effective enough to provide a meaningful and direct response to the patient's concerns."
    " Assess the coherence of the medical information provided, and focus on clarifying specific"
    " details relevant to the patient's inquiries. Each answer should be straightforward and directly"
    " address the medical questions posed, ensuring that it adds value to their understanding of the"
    " medical context. If the resource lacks a 'date', omit this detail from the reference."
)

# System prompt not using date and id
QUESTION_GEN_SYS_TMPL_2 = (
    "You are a healthcare expert specializing in patient communication. Your task is to create"
    " a clear and contextually accurate question from the perspective of the patient, as if you"
    " were the system embodying the patient, based on the provided flattened FHIR data. The question"
    " should address the patient's medical inquiries about their health condition, treatments, or"
    " any medical processes documented in the FHIR data. When crafting the answer, ensure it is"
    " effective enough to provide a meaningful and direct response to the patient's concerns."
    " Assess the coherence of the medical information provided, and focus on clarifying specific"
    " details relevant to the patient's inquiries. Each answer should be straightforward and directly"
    " address the medical questions posed, ensuring that it adds value to their understanding of the"
    " medical context. Omit any specific identifiers like dates or IDs from the question."
)

# User prompt
QUESTION_GEN_USER_TMPL = (
    "Context information is provided below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Based solely on the provided context information, "
    "generate a JSON object containing keys and values for relevant question and answer. "
    "Ensure that your response strictly follows this format: "
    '{{"questions_and_answers": ['
    '{{"question": "example question", "answer": "example answer"}}'
    "]}}."
    " If the context does not provide enough information to generate any question and answer, "
    "do not generate any output."
)


def create_directories(path_without_urls, path_with_urls, remove_urls):
    if remove_urls:
        if not os.path.exists(path_without_urls):
            os.makedirs(path_without_urls)
    else:
        if not os.path.exists(path_with_urls):
            os.makedirs(path_with_urls)
