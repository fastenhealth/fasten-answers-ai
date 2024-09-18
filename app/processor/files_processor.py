from datetime import datetime
import os
import csv


def csv_to_dict(file):
    decoded_file = file.decode("utf-8").splitlines()
    reader = csv.DictReader(decoded_file)
    return [dict(row) for row in reader]


def ensure_data_directory_exists():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_root, "data")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir


def generate_output_filename(process: str, task: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{process}_{task}_{timestamp}.csv"
