import json
import os


def load_params(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def create_directories(path_without_urls, path_with_urls, remove_urls):
    if remove_urls:
        if not os.path.exists(path_without_urls):
            os.makedirs(path_without_urls)
    else:
        if not os.path.exists(path_with_urls):
            os.makedirs(path_with_urls)


def create_flat_files_folder_if_not_exists(flat_file_path):
    if not os.path.exists(flat_file_path):
        os.mkdir(flat_file_path)
