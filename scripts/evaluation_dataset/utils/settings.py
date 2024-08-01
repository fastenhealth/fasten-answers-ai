from dotenv import load_dotenv
import logging
import os

from openai import OpenAI
import tiktoken


load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MAX_TOKENS = 2000

OPENAI_MODEL = "gpt-4o-mini"

OPENAI_MODEL_EMBEDDING = tiktoken.get_encoding("o200k_base")

QUESTION_GEN_SYS_TMPL = (
    "You are a knowledgeable and empathetic healthcare expert specializing in patient"
    " communication. Your task is to create {num_questions_per_chunk} patient-centered, clear,"
    " informative, and contextually accurate questions and answers based on provided FHIR data"
    " context. Ensure each answer is detailed, specific, and addresses all aspects of the patient's"
    " question. Avoid general statements and provide thorough explanations where necessary."
    " For example, if explaining a medical term or value, include what the specific number"
    " or range indicates, what it means for the patient's health, and any relevant next"
    " steps or considerations. If the specific value is not available in the provided context,"
    " indicate that the information is not present and suggest possible reasons or next steps."
    "\n"
    " Example:"
    "\n"
    " - If asked about a test result, include what the specific result value means,"
    " its normal range, and possible health implications."
    "\n"
    " - Original response: The glomerular filtration rate (GFR) test estimates how well"
    " your kidneys are filtering blood. It is a key indicator of kidney function and helps"
    " diagnose and monitor kidney disease."
    "\n"
    " - Improved response: The glomerular filtration rate (GFR) test estimates how well your"
    " kidneys are filtering blood. The value is adjusted for a body surface area of 1.73 square"
    " meters, which is an average adult size. A GFR of 90 or above is considered normal,"
    " while lower values can indicate varying degrees of kidney dysfunction. For example,"
    " a GFR between 60-89 may suggest mild kidney damage, whereas a GFR below 60 could indicate"
    " more severe impairment and the need for further investigation or treatment."
    "\n"
    " Use this approach to provide detailed and specific answers to patient questions."
)

QUESTION_GEN_USER_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "generate a JSON with keys and values of the relevant "
    "questions and answers, and do not add more information. "
    "Your response must be in this format: "
    "{{\"questions_and_answers\": ["
    "{{\"question\": \"example question\", \"answer\": \"example answer\"}}"
    "]}}"
)


client = OpenAI(
  api_key=OPENAI_API_KEY,
)
