# app/main.py
import sys
import tempfile
from pathlib import Path
from git import Repo

from vectorstore import VectorStore
from llm import get_chat_model
from ingest import load_repo_files
from graph import SummarizationGraph

def clone_github_repo(url):
    """Clone the GitHub repo into a temporary directory"""
    temp_dir = tempfile.TemporaryDirectory()
    repo_path = Path(temp_dir.name) / "repo"
    Repo.clone_from(url, repo_path, depth=1)
    return repo_path, temp_dir

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <github-url>")
        sys.exit(1)

    repo_url = sys.argv[1]

    # Clone repo
    repo_path, temp_dir = clone_github_repo(repo_url)

    # Load repo files as LangChain Documents
    docs = load_repo_files(repo_path)

    # Initialize FAISS VectorStore
    store = VectorStore()
    store.load()  # Attempt to load existing index
    if store.index is None:
        # Build if no index exists
        store.build_from_docs(docs)

    # Initialize local LLM (GGML GPT4All)
    llm = get_chat_model(model_name="ggml-gpt4all-j-v1.3-groovy", temperature=0)

    # Prepare state for multi-agent workflow
    state = {
        "query": "Summarize repository structure and explain architecture",
        "store": store,
        "llm": llm
    }

    # Run full agent graph
    graph = SummarizationGraph(state)
    result_state = graph.run()

    # Print final summary
    print("\n=== Repository Summary ===\n")
    print(result_state["result"])

    # Cleanup temporary repo clone
    temp_dir.cleanup()

if __name__ == "__main__":
    main()
