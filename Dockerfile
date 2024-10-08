FROM python:3.10-slim

WORKDIR /

COPY requirements.txt requirements.txt

COPY app /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
