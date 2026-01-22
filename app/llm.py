from langchain_community.llms import GPT4All
from langchain_core.prompts import PromptTemplate

def get_chat_model(model_name="ggml-gpt4all-j-v1.3-groovy", temperature=0):
    """
    Returns a local GPT4All LLM instance for offline usage.
    """
    return GPT4All(model=model_name, temperature=temperature)
