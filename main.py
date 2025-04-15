import os
import shutil
import subprocess
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl

# If you want to count tokens accurately with tiktoken:
try:
    import tiktoken
except ImportError:
    tiktoken = None  # Handle gracefully if tiktoken isn't installed

# Define extensions to auto-ignore silently.
IGNORED_EXTENSIONS = {".sqlite", ".png", ".jpeg"}

# Configuration
REFERENCE_DIR = Path(os.getenv("REFERENCE_DIR", "repos_reference"))
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Repo Manager API", description="Manage repos and file dumps for LLM context.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for prototype; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class RepoConfig(BaseModel):
    name: str
    url: HttpUrl

class FileDumpRequest(BaseModel):
    files: List[str]

def get_repo_path(repo_name: str) -> Path:
    return REFERENCE_DIR / repo_name

def run_git_command(command, cwd=None):
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Git command failed: {e.stderr.strip()}")

def get_file_tree(root: Path) -> dict:
    """
    Recursively builds a tree of all files/folders under 'root'.
    """
    tree = {"name": root.name, "children": []}
    # Sort by (is_file, name) so that folders come before files
    for entry in sorted(root.iterdir(), key=lambda x: (x.is_file(), x.name)):
        if entry.is_dir():
            tree["children"].append(get_file_tree(entry))
        else:
            tree["children"].append({"name": entry.name})
    return tree

def count_tokens_in_files(repo_path: Path, file_list: List[str]) -> int:
    """
    Count total tokens in the given file_list inside the repo_path using tiktoken.
    Files with extensions in IGNORED_EXTENSIONS will be silently ignored.
    """
    if tiktoken is None:
        raise HTTPException(status_code=500, detail="tiktoken is not installed on the server.")

    # GPT-3.5 and GPT-4 typically use the cl100k_base encoding
    encoder = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0
    for rel_path in file_list:
        file_path = repo_path / rel_path
        # Auto-ignore files with unsupported extensions.
        if file_path.suffix.lower() in IGNORED_EXTENSIONS:
            continue
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=400, detail=f"File not found: {rel_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            tokens = encoder.encode(content)
            total_tokens += len(tokens)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading file {rel_path}: {e}")
    return total_tokens

@app.get("/")
def index():
    """
    Return the index.html file located in the same directory as this server.
    """
    index_path = Path("index.html")
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(path=str(index_path), media_type="text/html")

@app.post("/repos", status_code=status.HTTP_201_CREATED)
def add_repo(config: RepoConfig):
    repo_path = get_repo_path(config.name)
    if repo_path.exists():
        raise HTTPException(status_code=400, detail="Repository already exists")
    try:
        run_git_command(["git", "clone", str(config.url), str(repo_path)])
        return {"message": f"Repository '{config.name}' added successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/repos/{repo_name}")
def remove_repo(repo_name: str):
    repo_path = get_repo_path(repo_name)
    if not repo_path.exists():
        raise HTTPException(status_code=404, detail="Repository not found")
    try:
        shutil.rmtree(repo_path)
        return {"message": f"Repository '{repo_name}' removed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/repos")
def list_repos():
    repos = [p.name for p in REFERENCE_DIR.iterdir() if p.is_dir()]
    return {"repositories": repos}

@app.post("/repos/{repo_name}/pull")
def pull_repo(repo_name: str):
    repo_path = get_repo_path(repo_name)
    if not repo_path.exists():
        raise HTTPException(status_code=404, detail="Repository not found")
    try:
        output = run_git_command(["git", "pull", "--ff-only"], cwd=str(repo_path))
        return {"message": f"Repository '{repo_name}' pulled successfully.", "details": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/repos/{repo_name}/tree")
def file_tree(repo_name: str):
    repo_path = get_repo_path(repo_name)
    if not repo_path.exists():
        raise HTTPException(status_code=404, detail="Repository not found")
    try:
        return get_file_tree(repo_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file tree: {e}")

@app.post("/repos/{repo_name}/dump")
def dump_files(repo_name: str, request: FileDumpRequest):
    repo_path = get_repo_path(repo_name)
    if not repo_path.exists():
        raise HTTPException(status_code=404, detail="Repository not found")

    concatenated_content = ""
    for rel_path in request.files:
        file_path = repo_path / rel_path
        # Auto-ignore files with unsupported extensions.
        if file_path.suffix.lower() in IGNORED_EXTENSIONS:
            continue
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=400, detail=f"File not found: {rel_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            concatenated_content += f"\n--- {rel_path} ---\n{content}\n"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading file {rel_path}: {e}")
    return {"concatenated": concatenated_content}

@app.get("/repos/{repo_name}/dump/all")
def dump_all_files(repo_name: str):
    """
    Return the concatenated contents of *every* file in the repo, 
    ignoring files with extensions defined in IGNORED_EXTENSIONS (e.g., .sqlite, .png, .jpeg),
    as well as the .git folder (to avoid infinite recursion), node_modules and __pycache__ directories.
    This endpoint returns all the readable file contents concatenated together.
    """
    repo_path = get_repo_path(repo_name)
    if not repo_path.exists():
        raise HTTPException(status_code=404, detail="Repository not found")

    concatenated_content = ""
    # Recursively walk the repository, skipping only the .git, node_modules, and __pycache__ directories
    for root, dirs, files in os.walk(repo_path):
        if ".git" in dirs:
            dirs.remove(".git")  # prevent walking into .git
        if "node_modules" in dirs:
            dirs.remove("node_modules")  # prevent walking into node_modules
        if "__pycache__" in dirs:
            dirs.remove("__pycache__")
        for filename in files:
            file_path = Path(root) / filename
            # Auto-ignore files with unsupported extensions.
            if file_path.suffix.lower() in IGNORED_EXTENSIONS:
                continue
            # Build a relative path for readability
            rel_path = file_path.relative_to(repo_path)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                concatenated_content += f"\n--- {rel_path} ---\n{content}\n"
            except Exception as e:
                # If it's a binary file or unreadable, handle gracefully:
                concatenated_content += f"\n--- {rel_path} (Unreadable) ---\nError: {e}\n"

    return {"concatenated": concatenated_content}

@app.get("/repos/{repo_name}/dump/auto-all")
def dump_auto_all_files(repo_name: str):
    """
    Return the concatenated contents of all files that are NOT ignored by:
      - .gitignore (if present)
      - IGNORED_EXTENSIONS
      - 'package-lock.json'
    Essentially the same as get_non_ignored_files, but we actually return file contents.
    """
    repo_path = get_repo_path(repo_name)
    if not repo_path.exists():
        raise HTTPException(status_code=404, detail="Repository not found")

    # If .gitignore is present, use 'git ls-files --others --cached --exclude-standard'
    gitignore_path = repo_path / ".gitignore"
    if gitignore_path.exists():
        command = ["git", "ls-files", "--others", "--cached", "--exclude-standard"]
        output = run_git_command(command, cwd=str(repo_path))
        file_list = output.splitlines()
    else:
        # If no .gitignore, just gather all files except the .git folder
        file_list = []
        for root, dirs, files in os.walk(repo_path):
            if ".git" in dirs:
                dirs.remove(".git")
            for f in files:
                abs_path = Path(root) / f
                rel_path = abs_path.relative_to(repo_path)
                file_list.append(str(rel_path))

    # Apply the existing auto-ignore logic
    # 1. skip any files with an extension in IGNORED_EXTENSIONS
    # 2. skip 'package-lock.json'
    # 3. skip any obviously unreadable/binary if you want to be extra safe
    file_list_filtered = []
    for f in file_list:
        if os.path.basename(f) == "package-lock.json":
            continue
        if Path(f).suffix.lower() in IGNORED_EXTENSIONS:
            continue
        file_list_filtered.append(f)

    concatenated_content = ""
    for rel_path_str in file_list_filtered:
        rel_path = Path(rel_path_str)
        file_path = repo_path / rel_path
        if file_path.is_file():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                concatenated_content += f"\n--- {rel_path} ---\n{content}\n"
            except Exception as e:
                concatenated_content += f"\n--- {rel_path} (Unreadable) ---\nError: {e}\n"

    return {"concatenated": concatenated_content}

@app.post("/repos/{repo_name}/tokens")
def calculate_tokens(repo_name: str, request: FileDumpRequest):
    repo_path = get_repo_path(repo_name)
    if not repo_path.exists():
        raise HTTPException(status_code=404, detail="Repository not found")

    total_tokens = count_tokens_in_files(repo_path, request.files)
    return {"total_tokens": total_tokens}

@app.get("/repos/{repo_name}/non_ignored_files")
def get_non_ignored_files(repo_name: str):
    """
    Return a flat list of all files in the repo that are *not* excluded by .gitignore.
    If there's no .gitignore, return all files in the repo.
    
    NEW: Additionally, auto-ignore any 'package-lock.json' files since they are rarely useful for LLMs.
    """
    repo_path = get_repo_path(repo_name)
    if not repo_path.exists():
        raise HTTPException(status_code=404, detail="Repository not found")

    gitignore_path = repo_path / ".gitignore"
    if gitignore_path.exists():
        # Use 'git ls-files' to get both tracked and untracked (non-ignored) files.
        command = ["git", "ls-files", "--others", "--cached", "--exclude-standard"]
        output = run_git_command(command, cwd=str(repo_path))
        file_list = output.splitlines()
    else:
        file_list = []
        for root, dirs, files in os.walk(repo_path):
            if ".git" in dirs:
                dirs.remove(".git")
            for f in files:
                abs_path = Path(root) / f
                rel_path = abs_path.relative_to(repo_path)
                file_list.append(str(rel_path))
    # NEW: Filter out any file whose basename is "package-lock.json"
    file_list = [f for f in file_list if os.path.basename(f) != "package-lock.json"]
    return {"files": file_list}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "32546"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
