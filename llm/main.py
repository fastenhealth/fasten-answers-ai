from fastapi import FastAPI

from fastapi.responses import StreamingResponse

from .llama import llm_client



app = FastAPI()

def stream_llm_response(query):

    def generate():
        for chunk in llm_client.chat(query):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/generate")
async def answer_query(query: str, k: int = 5, threshold: float = 0):


    return stream_llm_response(query)
