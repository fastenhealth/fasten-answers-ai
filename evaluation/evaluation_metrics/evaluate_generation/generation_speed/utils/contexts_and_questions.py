import random
import os
import pickle

from .settings import BASE_TEMPLATE, TEMPLATES, EXTRA_INFO_LIST, USER_QUESTIONS


def create_contexts_and_questions(
    tokenizer,
    templates: dict = TEMPLATES,
    base_template: str = BASE_TEMPLATE,
    extra_info_list: list = EXTRA_INFO_LIST,
    user_questions: list = USER_QUESTIONS,
) -> dict:
    contexts = {}
    questions = {}

    for target_token_size in range(100, 1600, 100):
        while True:
            question = random.choice(user_questions)
            # Base context
            context = base_template.format(
                age=random.randint(20, 80),
                gender=random.choice(templates["genders"]),
                conditions=", ".join(random.sample(templates["conditions_list"], random.randint(1, 3))),
                symptoms=", ".join(random.sample(templates["symptoms_list"], random.randint(1, 3))),
                medications=", ".join(random.sample(templates["medications_list"], random.randint(1, 3))),
                extra_info=random.choice(extra_info_list),
                user_question=question,
            )

            # Usar return_tensors=None para evitar PyTorch
            tokens = tokenizer.encode(context, return_tensors=None)
            token_count = len(tokens)

            # Increment context if necessary
            while token_count < target_token_size:
                context += " " + random.choice(extra_info_list)
                tokens = tokenizer.encode(context, return_tensors=None)  # Ajuste aquí también
                token_count = len(tokens)

            # Adjust context
            if token_count >= target_token_size - 10 and token_count <= target_token_size + 10:
                contexts[target_token_size] = context
                break

        contexts[target_token_size] = contexts[target_token_size].replace(question, "")
        questions[target_token_size] = question

    return contexts, questions


def create_contexts_and_questions_if_not_exist(
    tokenizer, contexts_path: str = "./data/contexts.pkl", questions_path: str = "./data/questions.pkl"
) -> dict:
    if os.path.exists(contexts_path) and os.path.exists(questions_path):
        with open(contexts_path, "rb") as f1, open(questions_path, "rb") as f2:
            contexts = pickle.load(f1)
            questions = pickle.load(f2)

    else:
        contexts, questions = create_contexts_and_questions(tokenizer=tokenizer)
        with open(contexts_path, "wb") as f1, open(questions_path, "wb") as f2:
            pickle.dump(contexts, f1)
            pickle.dump(questions, f2)

    return contexts, questions
