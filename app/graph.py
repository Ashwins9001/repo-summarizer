from langgraph.graph import StateGraph, START, END
from agents import retrieve_agent, summarize_agent, architect_agent

class SummarizationGraph:
    """
    Multi-agent graph workflow:
    1. Retrieve relevant docs
    2. Summarize docs
    3. Produce final architectural summary
    """
    def __init__(self, state: dict):
        self.state = state
        self.graph = StateGraph(state.__class__)  # Use the type of the initial state

        # Add nodes (nodes are just functions in LangGraph)
        self.graph.add_node("RetrieveDocs", self._retrieve)
        self.graph.add_node("SummarizeDocs", self._summarize)
        self.graph.add_node("ArchitectSummary", self._architect)

        # Define edges (linear execution)
        self.graph.add_edge(START, "RetrieveDocs")
        self.graph.add_edge("RetrieveDocs", "SummarizeDocs")
        self.graph.add_edge("SummarizeDocs", "ArchitectSummary")
        self.graph.add_edge("ArchitectSummary", END)

    # Internal wrappers to adapt agent functions
    def _retrieve(self, state: dict) -> dict:
        return retrieve_agent(state)

    def _summarize(self, state: dict) -> dict:
        return summarize_agent(state)

    def _architect(self, state: dict) -> dict:
        return architect_agent(state)

    def run(self):
        """
        Execute the full agent graph
        """
        compiled = self.graph.compile()
        # compiled is callable with the initial state
        result = compiled(self.state)
        return result
