"""
This script is based on the idea and some parts of the code from the repository "RAG_on_FHIR" by samschifman on GitHub.
Source: https://github.com/samschifman/RAG_on_FHIR.
Several modifications have been made to adapt it to this project's specific requirements.
"""

import base64
from io import BytesIO
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re

import PyPDF2


camel_pattern1 = re.compile(r"(.)([A-Z][a-z]+)")
camel_pattern2 = re.compile(r"([a-z0-9])([A-Z])")


def split_camel(text):
    new_text = camel_pattern1.sub(r"\1 \2", text)
    new_text = camel_pattern2.sub(r"\1 \2", new_text)
    return new_text


def handle_special_attributes(attrib_name, value):
    if attrib_name == "resource Type":
        return split_camel(value)
    return value


def flatten_fhir(nested_json):
    out = {}

    def flatten(json_to_flatten, name=""):
        if isinstance(json_to_flatten, dict):
            for sub_attribute in json_to_flatten:
                flatten(json_to_flatten[sub_attribute], name + split_camel(sub_attribute) + " ")
        elif isinstance(json_to_flatten, list):
            for i, sub_json in enumerate(json_to_flatten):
                flatten(sub_json, name + str(i) + " ")
        else:
            attrib_name = name[:-1]
            out[attrib_name] = handle_special_attributes(attrib_name, json_to_flatten)

    flatten(nested_json)
    return out


def flat_to_string(flat_entry):
    output = ""

    for attrib in flat_entry:
        output += f"{attrib} is {flat_entry[attrib]}. "

    return output


def measure_texts_lengths(file_path, section_lengths):
    plt.figure(figsize=(12, 3))
    plt.plot(section_lengths, marker="o")
    plt.title("Section lengths")
    plt.ylabel("# chars")
    plt.savefig(file_path)
    plt.close()


def extract_text_from_base64(content, content_type):
    try:
        # Decode the Base64 content
        decoded_data = base64.b64decode(content)

        # Extract the main content type (e.g., 'text/plain')
        main_content_type = content_type.split(";")[0].strip()

        if main_content_type == "application/pdf":
            # Extract text from PDF
            with BytesIO(decoded_data) as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = "".join(page.extract_text() for page in reader.pages)
            return text.replace("\n", " ")
        elif main_content_type == "text/plain":
            # Return plain text
            return decoded_data.decode("utf-8").replace("\n", " ")
        else:
            # Other content types can be handled here if needed
            return None  # Indicate unsupported or unprocessed content types
    except Exception as e:
        return f"Error decoding base64 content: {e}"


def extract_and_flatten_fhir(resource):
    flat_entry = flatten_fhir(resource)

    # Process DocumentReference and Binary resources for base64 content
    extracted_texts = []
    if resource.get("resourceType") == "DocumentReference":
        for content in resource.get("content", []):
            attachment = content.get("attachment", {})
            content_type = attachment.get("contentType", "")
            if "data" in attachment:
                # Decode base64 and extract text
                text = extract_text_from_base64(attachment["data"], content_type)
                if text:
                    extracted_texts.append(text)
    elif resource.get("resourceType") == "Binary":
        content_type = resource.get("contentType", "")
        text = extract_text_from_base64(resource.get("data", ""), content_type)
        if text:
            extracted_texts.append(text)

    # Replace base64 content with extracted text in the flattened entry
    if extracted_texts:
        # Merge all extracted texts into one single text representation for the flat entry
        flat_entry["extracted_text"] = " ".join(extracted_texts)

    return flat_entry


def flatten_bundle(bundle_file_name, flatten_files_path):
    file_name = bundle_file_name[bundle_file_name.rindex("/") + 1 : bundle_file_name.rindex(".")]

    with open(bundle_file_name) as raw:
        bundle = json.load(raw)
        resources_lengths = []
        total_bundle_entries = len(bundle["entry"])

        for i, entry in enumerate(bundle["entry"]):
            flat_entry = extract_and_flatten_fhir(entry["resource"])

            # Only write the extracted text if it exists, otherwise write the flat representation
            if "extracted_text" in flat_entry:
                text = flat_entry["extracted_text"]
            else:
                text = flat_to_string(flat_entry)

            len_text = len(text)
            resources_lengths.append(len_text)
            with open(f"{flatten_files_path}/{file_name}_{i}.txt", "w") as out_file:
                out_file.write(text)

    # Create plot for measure text lengths
    measure_texts_lengths("../data/texts_lengths.png", resources_lengths)

    new_flatten_files = os.listdir(flatten_files_path)

    return round(np.mean(resources_lengths), 2), total_bundle_entries, new_flatten_files
