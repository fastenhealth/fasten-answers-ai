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
    " communication. Your task is to create up to {num_questions_per_chunk} patient-centered,"
    " clear, informative, and contextually accurate questions and answers based only on the"
    " specific information provided in the FHIR data context. Focus on generating questions"
    " that explore specific details, implications, or next steps directly related to the"
    " context. Questions about medical terminology are valid if they clarify the implications"
    " or context-specific significance of a term for the patient. Do not generate questions"
    " that define statuses or classifications without context-specific relevance. Do not create"
    " questions about onset dates, abatement dates, encounter details, or the importance of"
    " recorded dates, regardless of their presence in the context. Prioritize questions that"
    " offer clear insights or value. Avoid making inferences or asking about information not"
    " explicitly present in the context. If the context does not provide enough detail to form"
    " a specific question or answer, do not create the question. Produce fewer questions or none"
    " at all if necessary. Generate less than 5 questions if there are not enough valuable questions"
    " to ask. Each answer should be detailed, specific, and address all aspects of the patient's question."
    " Provide thorough explanations where necessary."
    "\n"
    " Examples of valuable questions:"
    "\n"
    " - If asked about a test result, include what the specific result value means,"
    " its normal range, and possible health implications."
    "\n"
    " Types of questions that should not be asked:"
    "\n"
    " - What does 'abatement' mean in relation to my condition?"
    "\n"
    " - Can you explain the recorded date and its importance?"
    "\n"
    " - Is there any follow-up or monitoring required after the abatement of the condition?"
    "\n"
    " - Is there any follow-up needed after the abatement of the condition?"
    "\n"
    " - What does the abatement date signify in relation to my condition?"
    "\n"
    " - What does the onset date indicate about the duration of the condition?"
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
