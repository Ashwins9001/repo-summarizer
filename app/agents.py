from langchain.schema.messages import HumanMessage

def retrieve_agent(state):
    query = state["query"]
    store = state["store"]
    docs = store.similarity_search(query, k=6)
    return {**state, "docs": docs}

def summarize_agent(state):
    llm = state["llm"]
    text = "\n\n".join(
        f"{d.metadata['path']}:\n{d.page_content[:1500]}"
        for d in state["docs"]
    )
    summary = llm([
        HumanMessage(content=f"Summarize the following code:\n{text}")
    ])
    return {**state, "summary": summary.content}

def architect_agent(state):
    llm = state["llm"]
    final = llm([
        HumanMessage(
            content=f"""
You are a software architect.
Using the summary below, explain the repository architecture.

{state['summary']}
"""
        )
    ])
    return {**state, "result": final.content}
