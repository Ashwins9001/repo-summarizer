# Overview
Using the Facebook AI Similarity Search (FAISS) library to generate an embedding space that chunks repository code together and find semantically similar chunks. 

Multiagent system takes user-query, fetches data from FAISS to find relevant chunks, summarizes those chunks via summarizer-agent, and then asks an architect-agent to make meaning of a high-level structure. 

# Technology used
- LangGraph
- LangChain
- Llama 3.1 model
- FAISS vector store
- Docker