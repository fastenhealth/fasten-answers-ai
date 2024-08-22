from fastapi import FastAPI

from fastapi.responses import StreamingResponse

from utils.llm import LlamaCppClient
from utils.settings import LLAMA3_1_PROMPT, SYSTEM_PROMPT


import pdb

settings = {
    "host": "http://localhost:8080",
    "model_prompt": LLAMA3_1_PROMPT,
    "system_prompt": SYSTEM_PROMPT,
    "n_predict": 400,
    "temperature": 0.8,
    "stop": ["<|eot_id|>"],
    "stream": False
}

llm_client = LlamaCppClient(settings)


while True:
    user_prompt = input("Enter prompt: ")
    response = llm_client.chat(user_prompt=user_prompt)
    pdb.set_trace()

# app = FastAPI()

# def stream_llm_response(query):

#     def generate():
#         for chunk in llm_client.chat(query):
#             yield chunk

#     return StreamingResponse(generate(), media_type="text/plain")


# @app.get("/generate")
# async def answer_query(query: str, k: int = 5, threshold: float = 0):


#     return stream_llm_response(query)


# Todo:

"""
- configurar pruebas para que reciban los distintos prompts
- tomar los resultados de response.json() de tokens por segundo y todo eso
- Porbar con3 modelos cuantizados
"""
