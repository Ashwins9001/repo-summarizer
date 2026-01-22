import sys
from ingest import ingest_repo, clone_github_repo
from vectorstore import save_store, load_store
from llm import get_llm
from graph import build_graph

def ingest_local(path: str):
    docs = ingest_repo(path)
    save_store(docs)
    print(f"Ingested {len(docs)} files from {path}")

def ingest_github(url: str):
    repo_path, temp_dir = clone_github_repo(url)
    docs = ingest_repo(repo_path)
    save_store(docs)
    print(f"Ingested {len(docs)} files from {url}")
    temp_dir.cleanup()  # clean temporary clone

def run_graph(query: str):
    store = load_store()
    if store is None:
        print("No FAISS store found. Ingest first.")
        return
    llm = get_llm()
    graph = build_graph()
    result = graph.invoke({
        "query": query,
        "store": store,
        "llm": llm
    })
    print(result["result"])

def main():
    if len(sys.argv) < 2:
        print("Usage: main.py [ingest|ingest_github|summarize|ask] ...")
        return

    cmd = sys.argv[1]

    if cmd == "ingest":
        path = sys.argv[2]
        ingest_local(path)
    elif cmd == "ingest_github":
        url = sys.argv[2]
        ingest_github(url)
    elif cmd == "summarize":
        run_graph("Explain the repository")
    elif cmd == "ask":
        question = sys.argv[2]
        run_graph(question)
    else:
        print(f"Unknown command {cmd}")

if __name__ == "__main__":
    main()
