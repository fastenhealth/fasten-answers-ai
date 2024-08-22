import random

from settings import BASE_TEMPLATE, TEMPLATES, EXTRA_INFO_LIST, USER_QUESTIONS


def create_contexts(tokenizer,
                    templates: dict = TEMPLATES,
                    base_template: str = BASE_TEMPLATE,
                    extra_info_list: list = EXTRA_INFO_LIST,
                    user_questions: list = USER_QUESTIONS) -> dict:
    contexts = {}
    for target_token_size in range(100, 1600, 100):
        while True:
            # Base context
            context = base_template.format(
                age=random.randint(20, 80),
                gender=random.choice(templates["genders"]),
                conditions=", ".join(random.sample(templates["conditions_list"], random.randint(1, 3))),
                symptoms=", ".join(random.sample(templates["symptoms_list"], random.randint(1, 3))),
                medications=", ".join(random.sample(templates["medications_list"], random.randint(1, 3))),
                extra_info=random.choice(extra_info_list),
                user_question=random.choice(user_questions)
            )

            tokens = tokenizer.encode(context, return_tensors='pt')
            token_count = len(tokens[0])

            # Increment context if necessary
            while token_count < target_token_size:
                context += " " + random.choice(extra_info_list)
                tokens = tokenizer.encode(context, return_tensors='pt')
                token_count = len(tokens[0])

            # Adjust context
            if token_count >= target_token_size - 10 and token_count <= target_token_size + 10:
                contexts[target_token_size] = context
                break
    return contexts
