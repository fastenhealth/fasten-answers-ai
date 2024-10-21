import random
import pandas as pd

from evaluation.evaluation_metrics.evaluate_generation.generation_speed.utils.settings import (
    BASE_TEMPLATE,
    TEMPLATES,
    EXTRA_INFO_LIST,
    USER_QUESTIONS,
)


def create_contexts_and_questions(
    tokenizer,
    templates: dict = TEMPLATES,
    base_template: str = BASE_TEMPLATE,
    extra_info_list: list = EXTRA_INFO_LIST,
    user_questions: list = USER_QUESTIONS,
) -> dict:
    data = []

    for target_token_size in range(100, 300, 100):
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

            # USe return_tensors=None to avoid PyTorch
            tokens = tokenizer.encode(context, return_tensors=None)
            token_count = len(tokens)

            # Increment context if necessary
            while token_count < target_token_size:
                context += " " + random.choice(extra_info_list)
                tokens = tokenizer.encode(context, return_tensors=None)  # Ajuste aquí también
                token_count = len(tokens)

            # Adjust context
            if token_count >= target_token_size - 10 and token_count <= target_token_size + 10:
                context = context.replace(question, "")
                data.append((context, question, target_token_size))
                break

        df = pd.DataFrame(data, columns=["context", "question", "context_size"])

    return df
