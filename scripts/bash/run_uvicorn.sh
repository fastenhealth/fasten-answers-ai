#!/bin/bash

# Run Uvicorn server for FastAPI application
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
