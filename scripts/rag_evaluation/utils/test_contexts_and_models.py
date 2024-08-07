from dotenv import load_dotenv
import os
import random

from transformers import AutoTokenizer


load_dotenv()


base_template = (
    "Patient is a {age}-year-old {gender} with a history of {conditions}. "
    "They are currently experiencing {symptoms} and are taking {medications}. "
    "Additional information: {extra_info} "
    "Patient's question: {user_question}"
)

# Templates
genders = ["male", "female", "non-binary"]
conditions_list = ["hypertension", "diabetes", "asthma", "chronic kidney disease", "arthritis"]
symptoms_list = ["chest pain", "shortness of breath", "headache", "dizziness", "fatigue"]
medications_list = ["lisinopril", "metformin", "albuterol", "hydrochlorothiazide", "ibuprofen"]

# Extra information
extra_info_list = [
    "The patient has a family history of cardiovascular disease.",
    "They have been experiencing symptoms for the past two weeks.",
    "Regular follow-ups are scheduled every month.",
    "The patient follows a specific diet plan and exercises regularly.",
    "Their last lab results showed improvement in glucose levels.",
    "The patient has been advised to monitor their blood pressure daily.",
    "Recent imaging studies indicate no acute changes.",
    "The patient has been vaccinated for seasonal influenza.",
    "They reported a recent increase in physical activity.",
    "Current lifestyle includes smoking cessation and reduced alcohol intake.",
    "Previous surgical history includes appendectomy at age 25.",
    "The patient has mild allergic reactions to certain antibiotics.",
    "They have been on a gluten-free diet for the past year.",
    "The patient's occupation involves moderate physical labor.",
    "Recent laboratory tests revealed elevated cholesterol levels.",
    "They have regular appointments with a nutritionist.",
    "The patient is participating in a clinical trial for new medication.",
    "Their sleep patterns have improved with recent lifestyle changes.",
    "They have joined a support group for chronic pain management.",
    "The patient uses a home blood pressure monitor to track readings.",
    "They have expressed interest in alternative therapies.",
    "Their exercise regimen includes daily walking and stretching exercises.",
    "The patient has reported improved mental health with therapy.",
    "They are managing stress with mindfulness and meditation practices.",
    "The patient has completed a course on diabetes self-management.",
]

# Users questions
user_questions = [
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


llama_prompt = ("A chat between a curious user and an intelligent, "
                "polite medical assistant. The assistant provides detailed, "
                "helpful answers to the user's medical questions, "
                "including accurate references where applicable.")

def create_contexts(tokenizer):
    contexts = {}
    for target_token_size in range(100, 1600, 100):
        while True:
            # Genera un contexto base
            context = base_template.format(
                age=random.randint(20, 80),
                gender=random.choice(genders),
                conditions=", ".join(random.sample(conditions_list, random.randint(1, 3))),
                symptoms=", ".join(random.sample(symptoms_list, random.randint(1, 3))),
                medications=", ".join(random.sample(medications_list, random.randint(1, 3))),
                extra_info=random.choice(extra_info_list),
                user_question=random.choice(user_questions)
            )
            
            # Tokeniza el contexto
            tokens = tokenizer.encode(context, return_tensors='pt')
            token_count = len(tokens[0])

            # Incrementa el contexto si es necesario
            while token_count < target_token_size:
                context += " " + random.choice(extra_info_list)
                tokens = tokenizer.encode(context, return_tensors='pt')
                token_count = len(tokens[0])
            
            # Ajusta para no exceder el tamaño objetivo
            if token_count >= target_token_size - 10 and token_count <= target_token_size + 10:
                contexts[target_token_size] = context
                break
    return contexts


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B",
                                            token=os.getenv("HUGGING_FACE_ACCESS_TOKEN"))
    contexts = create_contexts(tokenizer)

