from langgraph.graph import StateGraph
from agents import retrieve_agent, summarize_agent, architect_agent

'''
Each node is a function that transforms shared state
Nodes are combined via edges to define a DAG
Define multiagent orchestration as DAG

State is defined as following dict:

state = {
    "query": "...",
    "store": FAISS(...),
    "llm": ChatOllama(...),
    "docs": [...],
    "summary": "...",
    "result": "..."
}

'''

def build_graph():
    graph = StateGraph(dict)
    graph.add_node("retrieve", retrieve_agent)
    graph.add_node("summarize", summarize_agent)
    graph.add_node("architect", architect_agent)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "summarize")
    graph.add_edge("summarize", "architect")
    return graph.compile()