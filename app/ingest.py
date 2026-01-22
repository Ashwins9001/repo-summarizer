from langchain.docstore.document import Document
from git import Repo
from pathlib import Path
import tempfile

def ingest_repo(repo_path: str):
    docs = []
    repo_path = Path(repo_path)
    for file in repo_path.rglob("*"):
        if file.is_file() and file.suffix in {".py", ".md", ".ts", ".js"}:
            try:
                content = file.read_text(errors="ignore")
                docs.append(
                    Document(
                        page_content=content,
                        metadata={"path": str(file.relative_to(repo_path))}
                    )
                )
            except Exception:
                pass
    return docs

def clone_github_repo(url: str):
    temp_dir = tempfile.TemporaryDirectory()
    repo_path = Path(temp_dir.name)
    print(f"Cloning {url} into {repo_path}")
    Repo.clone_from(url, repo_path, depth=1)
    return repo_path, temp_dir
