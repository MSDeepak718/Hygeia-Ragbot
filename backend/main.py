from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from query import RagBot

app = FastAPI()
chatbot = RagBot()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    
@app.post("/api/chat")
def chat(query: Query):
    response = chatbot.ask(query.question)
    return {"answer": response}