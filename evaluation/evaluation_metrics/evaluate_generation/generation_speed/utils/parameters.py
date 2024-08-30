import json


def load_params(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)
