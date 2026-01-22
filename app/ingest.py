from pathlib import Path
from langchain_core.documents import Document

def load_repo_files(repo_path: Path, exts=(".py", ".md", ".txt")) -> list[Document]:
    """
    Reads all files in the repo with given extensions and returns as LangChain Documents
    """
    docs = []
    for file in repo_path.rglob("*"):
        if file.suffix.lower() in exts:
            try:
                content = file.read_text(encoding="utf-8")
                docs.append(Document(page_content=content, metadata={"path": str(file)}))
            except Exception:
                continue
    return docs
