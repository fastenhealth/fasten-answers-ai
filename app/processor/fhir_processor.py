import base64
from io import BytesIO
import json
import re

import PyPDF2


def read_json_FHIR(json_path):
    with open(json_path, "r") as f:
        fhir_data = json.load(f)
    return fhir_data


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


def extract_text(resource):
    """
    Process DocumentReference, Binary, DiagnosticReport, and Observation resources for base64 content
    """
    if resource.get("resourceType") == "DocumentReference":
        for content in resource.get("content", []):
            attachment = content.get("attachment", {})
            content_type = attachment.get("contentType", "")
            if "data" in attachment and "text/" in content_type:
                # Decode base64 and extract text
                text = extract_text_from_base64(
                    attachment["data"], content_type)
                if text:
                    attachment["data"] = text
    elif resource.get("resourceType") == "Binary":
        content_type = resource.get("contentType", "")
        if "text/" in content_type or "application/pdf" in content_type:
            text = extract_text_from_base64(
                resource.get("data", ""), content_type)
            if text:
                resource["data"] = text
    elif resource.get("resourceType") == "DiagnosticReport":
        for form in resource.get("presentedForm", []):
            content_type = form.get("contentType", "")
            if "data" in form and "text/" in content_type:
                # Decode base64 and extract text
                text = extract_text_from_base64(form["data"], content_type)
                if text:
                    form["data"] = text
    elif resource.get("resourceType") == "Observation":
        if "valueAttachment" in resource:
            attachment = resource.get("valueAttachment", {})
            content_type = attachment.get("contentType", "")
            if "data" in attachment and "text/" in content_type:
                # Decode base64 and extract text
                text = extract_text_from_base64(
                    attachment["data"], content_type)
                if text:
                    attachment["data"] = text

    return resource


def remove_urls_from_fhir(data):
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")

    if isinstance(data, dict):
        return {key: remove_urls_from_fhir(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [remove_urls_from_fhir(item) for item in data]
    elif isinstance(data, str):
        return url_pattern.sub("", data)
    else:
        return data


def process_resources(data: dict,
                      remove_urls: bool) -> list[dict]:
    resources = []

    for entry in data.get("entry", []):
        if "resource" in entry:
            resource = entry["resource"]
            resource_type = resource.get("resourceType")

            resource = extract_text(resource)

            if remove_urls:
                resource = remove_urls_from_fhir(resource)

            resource_id = resource.get("id")
            text = json.dumps(resource).replace("\\", "")

            resources.append({"resource_id": resource_id,
                              "resource_type": resource_type,
                              "resource": text})

    return resources
