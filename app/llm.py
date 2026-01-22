from langchain_community.chat_models import ChatOllama
import os

def get_llm():
    return ChatOllama(
        model="llama3.1",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
        temperature=0.2,
    )
