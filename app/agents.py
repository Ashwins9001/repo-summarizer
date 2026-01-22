from langchain.messages import HumanMessage

def retrieve_agent(state: dict) -> dict:
    query = state["query"]
    store = state["store"]
    docs = store.search(query, k=6)
    return {**state, "docs": docs}

def summarize_agent(state: dict) -> dict:
    llm = state["llm"]
    text = "\n\n".join(
        f"{d.metadata.get('path', 'unknown')}:\n{d.page_content[:1500]}"
        for d in state["docs"]
    )
    summary_msg = llm([HumanMessage(content=f"Summarize the following code:\n{text}")])
    return {**state, "summary": summary_msg.content}

def architect_agent(state: dict) -> dict:
    llm = state["llm"]
    final_msg = llm([HumanMessage(
        content=f"You are a software architect.\nUsing the summary below, explain the repository architecture:\n\n{state['summary']}"
    )])
    return {**state, "result": final_msg.content}
