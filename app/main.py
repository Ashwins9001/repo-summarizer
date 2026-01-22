import sys
from ingest import ingest_repo, clone_github_repo
from vectorstore import save_store, load_store
from llm import get_llm
from graph import build_graph

def ingest_and_run_github(url: str):
    # Clone GitHub repo
    repo_path, temp_dir = clone_github_repo(url)
    
    # Ingest files
    docs = ingest_repo(repo_path)
    store = save_store(docs)
    
    print(f"Ingested {len(docs)} files from {url}")
    
    # Run summarization graph
    llm = get_llm()
    graph = build_graph()
    result = graph.invoke({
        "query": "Explain the repository",
        "store": store,
        "llm": llm
    })
    
    print("\n===== Repository Summary =====\n")
    print(result["result"])
    
    # Cleanup temporary clone
    temp_dir.cleanup()

def main():
    if len(sys.argv) < 3:
        print("Usage: main.py github <GitHub-URL>")
        return
    
    cmd = sys.argv[1]
    if cmd == "github":
        url = sys.argv[2]
        ingest_and_run_github(url)
    else:
        print(f"Unknown command: {cmd}")

if __name__ == "__main__":
    main()
