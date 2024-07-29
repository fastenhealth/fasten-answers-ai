from utils.parameters import load_params
from utils.llm_client import configure_llm_client
from utils.generator import generate_responses
from utils.contexts_and_questions import create_contexts_and_questions_if_not_exist


def main():
    params = load_params("./data/input.json")

    llm_client = configure_llm_client(params)

    contexts, questions = create_contexts_and_questions_if_not_exist(llm_client.tokenizer)

    generate_responses(params, contexts, questions, llm_client)


if __name__ == '__main__':
    main()
