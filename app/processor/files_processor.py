import csv


def csv_to_dict(file):
    decoded_file = file.decode("utf-8").splitlines()
    reader = csv.DictReader(decoded_file)
    return [dict(row) for row in reader]
