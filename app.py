from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import uvicorn

app = FastAPI()

# CORS middleware setup
app.add_middleware(CORSMiddleware,
                 allow_origins=["*"],  # Update this to your allowed origins in production
                 allow_credentials=True,
                 allow_methods=["*"],
                 allow_headers=["*"])

# Model to handle the chat message
class ChatMessage(BaseModel):
    message: str

# In-memory chat history
chat_history: List[str] = []

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Info endpoint
@app.get("/info")
async def info():
    return {"app": "DovIA-LLM FastAPI", "version": "1.0.0"}

# Chat endpoint
@app.post("/chat")
async def chat(message: ChatMessage):
    chat_history.append(message.message)
    response = await get_response(message.message)
    return {"response": response, "history": chat_history}

async def get_response(message: str):
    # Simulate a response delay (e.g. loading a model)
    await asyncio.sleep(1)
    return f"Echo: {message}"

# Streaming response example
@app.post("/stream_chat")
async def stream_chat(message: ChatMessage):
    async def event_stream():
        for msg in chat_history:
            yield msg   # This assumes messages are streamed back
            await asyncio.sleep(1) # Simulate delay to mimic streaming

    return StreamingResponse(event_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
