from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os

# Initialize FastAPI app
app = FastAPI(title="Simple FastAPI with Groq Chat Completion")

# Set up the OpenAI (Groq) client using new SDK interface
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY", "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"),
    base_url="https://api.groq.com/openai/v1"
)

# Pydantic model for chat request
class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 100

# Hello World route
@app.get("/")
async def hello_world():
    return {"message": "Hello, World! Welcome to FastAPI with Groq!"}

# Fun route to play around with FastAPI
@app.get("/greet/{name}")
async def greet_user(name: str):
    return {"greeting": f"Hey {name}! Thanks for visiting our FastAPI server!"}

# Chat completion route using Groq
@app.post("/chat")
async def chat_completion(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # Groq-supported model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": request.prompt}
            ],
            max_tokens=request.max_tokens,
            temperature=0.7
        )
        return {"response": response.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")

# Run the server with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
