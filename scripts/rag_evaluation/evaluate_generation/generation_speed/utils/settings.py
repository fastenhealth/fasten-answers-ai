from dotenv import load_dotenv
import os

from transformers import AutoTokenizer


load_dotenv()


SYSTEM_PROMPT = ("A chat between a curious user and an intelligent, "
                 "polite medical assistant. The assistant provides detailed, "
                 "helpful answers to the user's medical questions, "
                 "including accurate references where applicable. "
                 "The user will give you context and then his/her "
                 "question will come in."
                 )

LLAMA3_PROMPT = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{system_prompt}<|eot_id|>\n"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Context information is below.\n "
    "---------------------\n "
    "{user_prompt}\n"
    "---------------------\n "
    "Given the context information (if there is any), "
    "this is my message: "
    "{question}<|eot_id|>\n"
    "<|start_header_id|>assistant<|end_header_id|>"
)

PHI_3_5_PROMPT = (
    "<|system|>"
    "{system_prompt}<|end|>"
    "<|user|>"
    "Context information is below.\n "
    "---------------------\n "
    "{user_prompt}\n"
    "---------------------\n "
    "{question}"
    "<|end|>"
    "<|assistant|>"
)

# Base template
BASE_TEMPLATE = (
    "Patient is a {age}-year-old {gender} with a history of {conditions}. "
    "The patient is currently experiencing {symptoms} and is taking {medications}. "
    "Additional information: {extra_info} \n"
    "{user_question}"
)

# Templates
TEMPLATES = {
    "genders": ["male", "female", "non-binary"],
    "conditions_list": ["hypertension", "diabetes", "asthma", "chronic kidney disease", "arthritis"],
    "symptoms_list": ["chest pain", "shortness of breath", "headache", "dizziness", "fatigue"],
    "medications_list": ["lisinopril", "metformin", "albuterol", "hydrochlorothiazide", "ibuprofen"]
}
# Extra information
EXTRA_INFO_LIST = [
    "The patient has a family history of cardiovascular disease.",
    "They have been experiencing symptoms for the past two weeks.",
    "Regular follow-ups are scheduled every month.",
    "The patient follows a specific diet plan and exercises regularly.",
    "Their last lab results showed improvement in glucose levels.",
    "The patient has been advised to monitor their blood pressure daily.",
    "Recent imaging studies indicate no acute changes.",
    "The patient has been vaccinated for seasonal influenza.",
    "The patient reported a recent increase in physical activity.",
    "Current lifestyle includes smoking cessation and reduced alcohol intake.",
    "Previous surgical history includes appendectomy at age 25.",
    "The patient has mild allergic reactions to certain antibiotics.",
    "The patient have been on a gluten-free diet for the past year.",
    "The patient's occupation involves moderate physical labor.",
    "Recent laboratory tests revealed elevated cholesterol levels.",
    "The patient have regular appointments with a nutritionist.",
    "The patient is participating in a clinical trial for new medication.",
    "The patient sleep patterns have improved with recent lifestyle changes.",
    "The patient have joined a support group for chronic pain management.",
    "The patient uses a home blood pressure monitor to track readings.",
    "The patient have expressed interest in alternative therapies.",
    "The patient exercise regimen includes daily walking and stretching exercises.",
    "The patient has reported improved mental health with therapy.",
    "The patient is managing stress with mindfulness and meditation practices.",
    "The patient has completed a course on diabetes self-management.",
]

# Users questions
USER_QUESTIONS = [
    "What should I do to better manage my symptoms?",
    "Are there any alternative therapies I should consider?",
    "How can I adjust my diet to improve my condition?",
    "What exercises would be beneficial for my health?",
    "Should I be concerned about my recent lab results?",
    "How often should I schedule follow-up appointments?",
    "Is my current medication regimen appropriate?",
    "What are the potential side effects of my medications?",
    "How can I reduce my risk of complications?",
    "Are there any new treatments available for my condition?",
]


MODEL_SETTINGS = {"meta-llama/Meta-Llama-3.1-8B-Instruct":
                  {"tokenizer": AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B",
                                                              token=os.getenv("HUGGING_FACE_ACCESS_TOKEN")),
                   "model_prompt": LLAMA3_PROMPT
                   },
                  "microsoft/Phi-3.5-mini-instruct":
                  {"tokenizer": AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct",
                                                              trust_remote_code=True),
                      "model_prompt": PHI_3_5_PROMPT
                   }
                  }
