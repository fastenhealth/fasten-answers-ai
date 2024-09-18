FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt

# Copy the entire app folder inside the app folder in the workdir
COPY /app app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
