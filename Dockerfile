FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/

EXPOSE 8000
