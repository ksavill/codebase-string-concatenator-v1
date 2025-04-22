import os
import shutil
import subprocess
import json
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Set up logging for detailed debugging output.
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

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

# Metadata file to save repository configuration (git vs local and original URL/path)
METADATA_FILE = REFERENCE_DIR / ".repo_metadata.json"

def load_metadata() -> dict:
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                logging.debug(f"Loaded metadata: {metadata}")
                return metadata
        except Exception as e:
            logging.error(f"Error loading metadata: {e}")
            return {}
    return {}

def save_metadata(metadata: dict):
    try:
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
        logging.debug("Metadata saved successfully.")
    except Exception as e:
        logging.error(f"Error saving metadata: {e}")

app = FastAPI(
    title="Repo Manager API",
    description="Manage repos (Git or local) and file dumps for LLM context."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for prototype; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FileDumpRequest(BaseModel):
    files: List[str]

class RepoConfig(BaseModel):
    name: str
    url: str
    repo_type: str = "git"
    branch: Optional[str] = None  # New optional field

def get_repo_path(repo_name: str) -> Path:
    path = REFERENCE_DIR / repo_name
    logging.debug(f"Computed repo path for {repo_name}: {path}")
    return path

def run_git_command(command, cwd=None):
    logging.debug(f"Running git command: {' '.join(command)} in {cwd}")
    try:
        result = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, check=True
        )
        logging.debug(f"Git command output: {result.stdout.strip()}")
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Git command error: {e.stderr.strip()}")
        raise HTTPException(
            status_code=500, detail=f"Git command failed: {e.stderr.strip()}"
        )

def get_file_tree(root: Path, visited: set = None) -> dict:
    """
    Recursively builds a tree of all files/folders under 'root' with logging.
    Cycle detection is used to avoid infinite recursion in case of symlink loops.
    Also ignores any directory named 'repos_reference' to avoid self recursion.
    """
    if visited is None:
        visited = set()
    
    try:
        real_path = root.resolve()
    except Exception as e:
        logging.error(f"Error resolving path {root}: {e}")
        return {"name": root.name, "children": []}

    if real_path in visited:
        logging.warning(f"Cycle detected: {real_path} already visited. Skipping directory: {root}")
        return {"name": root.name, "children": []}
    visited.add(real_path)

    # If the directory is named 'repos_reference' and it's not the root itself, ignore it.
    if root.name == "repos_reference" and root != REFERENCE_DIR:
        logging.debug(f"Ignoring directory '{root}' as it is hardcoded to be skipped.")
        return {"name": root.name, "children": []}

    logging.debug(f"Processing directory: {root} (real: {real_path})")
    tree = {"name": root.name, "children": []}
    try:
        for entry in sorted(root.iterdir(), key=lambda x: (x.is_file(), x.name)):
            # Skip the repos_reference directory to avoid infinite recursion.
            if entry.is_dir() and entry.name == "repos_reference":
                logging.debug(f"Skipping subdirectory '{entry}' as it is 'repos_reference'.")
                continue
            if entry.is_dir():
                tree["children"].append(get_file_tree(entry, visited))
            else:
                logging.debug(f"Found file: {entry}")
                tree["children"].append({"name": entry.name})
        return tree
    except Exception as e:
        logging.error(f"Error processing directory {root}: {e}")
        return {"name": root.name, "children": []}

def count_tokens_in_files(repo_path: Path, file_list: List[str]) -> int:
    """
    Count total tokens in the given file_list inside the repo_path using tiktoken.
    Silently skips any file that:
      - has an extension in IGNORED_EXTENSIONS
      - cannot be decoded as UTF-8 (i.e., is non-text/binary)
    """
    if tiktoken is None:
        raise HTTPException(
            status_code=500,
            detail="tiktoken is not installed on the server."
        )

    encoder = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0
    for rel_path in file_list:
        file_path = repo_path / rel_path

        # skip known binary extensions outright
        if file_path.suffix.lower() in IGNORED_EXTENSIONS:
            logging.debug(f"Skipping ignored extension: {file_path}")
            continue

        if not file_path.exists() or not file_path.is_file():
            logging.error(f"File not found: {rel_path}")
            raise HTTPException(status_code=400, detail=f"File not found: {rel_path}")

        # try reading as UTF-8; skip if it fails
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            logging.debug(f"Skipping non-text file during token count: {file_path} ({e})")
            continue
        except Exception as e:
            logging.error(f"Error reading file {rel_path}: {e}")
            # raise HTTPException(status_code=500, detail=f"Error reading file {rel_path}: {e}")

        tokens = encoder.encode(content)
        logging.debug(f"Counted {len(tokens)} tokens for {file_path}")
        total_tokens += len(tokens)

    return total_tokens

@app.get("/")
def index():
    """
    Return the index.html file located in the same directory as this server.
    """
    index_path = Path("index.html")
    logging.debug("Serving index.html")
    if not index_path.exists():
        logging.error("index.html not found.")
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(path=str(index_path), media_type="text/html")

@app.post("/repos", status_code=status.HTTP_201_CREATED)
def add_repo(config: RepoConfig):
    """
    Add a new 'repository', which could be:
    - A Git repository (cloned from a remote URL),
    - A local directory (repo_type='local').
    """
    logging.info(f"Adding repository '{config.name}' of type '{config.repo_type}'")
    repo_path = get_repo_path(config.name)
    if repo_path.exists():
        logging.error(f"Repository '{config.name}' already exists.")
        raise HTTPException(status_code=400, detail="Repository/directory already exists")

    if config.repo_type == "git":
        try:
            command = ["git", "clone"]
            if config.branch:
                command += ["-b", config.branch]
            command += [str(config.url), str(repo_path)]
            run_git_command(command)
            metadata = load_metadata()
            metadata[config.name] = {
                "repo_type": "git",
                "url": config.url,
                "branch": config.branch or "default"
            }
            save_metadata(metadata)
            logging.info(f"Git repository '{config.name}' added successfully.")
            return {"message": f"Git repository '{config.name}' added successfully."}
        except Exception as e:
            logging.error(f"Error adding git repository '{config.name}': {e}")
            raise HTTPException(status_code=500, detail=str(e))
    elif config.repo_type == "local":
        local_path = Path(config.url).expanduser().resolve()
        if not local_path.is_dir():
            logging.error(f"Local path '{config.url}' does not exist or is not a directory.")
            raise HTTPException(
                status_code=400,
                detail=f"Local path '{config.url}' does not exist or is not a directory."
            )
        try:
            os.symlink(local_path, repo_path, target_is_directory=True)
            method_used = "symlink"
            logging.info(f"Local directory '{config.name}' added via symlink.")
        except OSError as e:
            logging.warning(f"Symlink failed for '{config.name}': {e}. Attempting copy.")
            try:
                shutil.copytree(local_path, repo_path)
                method_used = "copy"
                logging.info(f"Local directory '{config.name}' added via copy.")
            except Exception as e:
                logging.error(f"Error copying local directory '{config.name}': {e}")
                raise HTTPException(status_code=500, detail=str(e))
        metadata = load_metadata()
        metadata[config.name] = {"repo_type": "local", "url": str(local_path), "method": method_used}
        save_metadata(metadata)
        return {"message": f"Local directory '{config.name}' added successfully via {method_used}."}
    else:
        logging.error("Invalid repo_type provided.")
        raise HTTPException(
            status_code=400,
            detail="Invalid repo_type. Must be 'git' or 'local'."
        )

@app.delete("/repos/{repo_name}")
def remove_repo(repo_name: str):
    """
    Remove a repository or local directory (which was either cloned or symlinked/copied).
    """
    logging.info(f"Removing repository '{repo_name}'.")
    repo_path = get_repo_path(repo_name)
    if not repo_path.exists():
        logging.error(f"Repository '{repo_name}' not found.")
        raise HTTPException(status_code=404, detail="Repository/directory not found")
    try:
        if repo_path.is_symlink():
            repo_path.unlink()
            logging.debug("Removed symlink.")
        else:
            shutil.rmtree(repo_path)
            logging.debug("Removed directory recursively.")
        metadata = load_metadata()
        if repo_name in metadata:
            del metadata[repo_name]
            save_metadata(metadata)
        logging.info(f"Repository '{repo_name}' removed successfully.")
        return {"message": f"Repository/Directory '{repo_name}' removed successfully."}
    except Exception as e:
        logging.error(f"Error removing repository '{repo_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/repos")
def list_repos():
    """
    Returns all items (Git or local) that exist in repos_reference along with attributes.
    """
    logging.debug("Listing all repositories.")
    metadata = load_metadata()
    repos = []
    for p in REFERENCE_DIR.iterdir():
        if not (p.is_dir() or p.is_symlink()):
            continue
        if p.name.startswith("."):
            continue
        repo_meta = metadata.get(p.name)
        if repo_meta:
            repo_type = repo_meta.get("repo_type", "local")
        else:
            repo_type = "git" if (p / ".git").exists() else "local"
        repos.append({
            "name": p.name,
            "repo_type": repo_type,
            "branch": metadata.get(p.name, {}).get("branch", "default")
        })
    logging.debug(f"Found repositories: {repos}")
    return {"repositories": repos}

@app.post("/repos/{repo_name}/pull")
def pull_repo(repo_name: str):
    """
    Attempt to pull from remote if it's a Git repository. For local repositories,
    refresh (update) the contents if possible.
    """
    logging.info(f"Pull/refresh repository '{repo_name}'.")
    repo_path = get_repo_path(repo_name)
    if not repo_path.exists():
        logging.error(f"Repository '{repo_name}' not found.")
        raise HTTPException(status_code=404, detail="Repository/directory not found")

    metadata = load_metadata()
    repo_meta = metadata.get(repo_name)
    if repo_meta and repo_meta.get("repo_type") == "local":
        if repo_path.is_symlink():
            target = os.readlink(repo_path)
            logging.info(f"Local repository '{repo_name}' is a symlink to '{target}', no refresh needed.")
            return {
                "message": f"Local directory '{repo_name}' is a symlink to '{target}'. No refresh needed."
            }
        else:
            source = repo_meta.get("url")
            if not source:
                logging.error("Metadata for local repository is missing source path.")
                raise HTTPException(status_code=500, detail="Metadata for local repository is missing source path.")
            source_path = Path(source).expanduser().resolve()
            if not source_path.is_dir():
                logging.error(f"Source local directory '{source}' does not exist.")
                raise HTTPException(status_code=400, detail=f"Source local directory '{source}' does not exist.")
            try:
                logging.debug(f"Refreshing local directory '{repo_name}' from source '{source}'.")
                shutil.rmtree(repo_path)
                shutil.copytree(source_path, repo_path)
                logging.info(f"Local directory '{repo_name}' refreshed successfully.")
                return {"message": f"Local directory '{repo_name}' refreshed successfully from source: {source}."}
            except Exception as e:
                logging.error(f"Error refreshing local repository '{repo_name}': {e}")
                raise HTTPException(status_code=500, detail=f"Error refreshing local repository: {e}")
    else:
        git_folder = repo_path / ".git"
        if not git_folder.exists():
            logging.warning("No .git folder found; assuming local directory. No refresh performed.")
            return {
                "message": "No .git folder found; assuming local directory. No refresh performed."
            }
        try:
            output = run_git_command(["git", "pull", "--ff-only"], cwd=str(repo_path))
            logging.info(f"Repository '{repo_name}' pulled successfully.")
            return {
                "message": f"Repository '{repo_name}' pulled successfully.",
                "details": output
            }
        except Exception as e:
            logging.error(f"Error pulling git repository '{repo_name}': {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/repos/{repo_name}/tree")
def file_tree(repo_name: str):
    logging.info(f"Building file tree for repository '{repo_name}'.")
    repo_path = get_repo_path(repo_name)
    if not repo_path.exists():
        logging.error(f"Repository '{repo_name}' not found.")
        raise HTTPException(status_code=404, detail="Repository/directory not found")
    try:
        tree = get_file_tree(repo_path)
        logging.debug(f"File tree for '{repo_name}': {tree}")
        return tree
    except Exception as e:
        logging.error(f"Error reading file tree for repository '{repo_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Error reading file tree: {e}")

@app.post("/repos/{repo_name}/dump")
def dump_files(repo_name: str, request: FileDumpRequest):
    logging.info(f"Dumping selected files for repository '{repo_name}'.")
    repo_path = get_repo_path(repo_name)
    if not repo_path.exists():
        logging.error(f"Repository '{repo_name}' not found.")
        raise HTTPException(status_code=404, detail="Repository/directory not found")

    concatenated_content = ""
    for rel_path in request.files:
        file_path = repo_path / rel_path
        if file_path.suffix.lower() in IGNORED_EXTENSIONS:
            logging.debug(f"Skipping ignored file: {rel_path}")
            continue
        if not file_path.exists() or not file_path.is_file():
            logging.error(f"File not found: {rel_path}")
            raise HTTPException(status_code=400, detail=f"File not found: {rel_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            concatenated_content += f"\n--- {rel_path} ---\n{content}\n"
            logging.debug(f"Appended content from file: {rel_path}")
        except Exception as e:
            logging.error(f"Error reading file {rel_path}: {e}")
            # raise HTTPException(status_code=500, detail=f"Error reading file {rel_path}: {e}")
    return {"concatenated": concatenated_content}

@app.get("/repos/{repo_name}/dump/all")
def dump_all_files(repo_name: str):
    """
    Return the concatenated contents of every file in the repo/directory,
    ignoring:
    - .git folder
    - node_modules
    - __pycache__
    - any file extension in IGNORED_EXTENSIONS
    """
    logging.info(f"Dumping all files for repository '{repo_name}'.")
    repo_path = get_repo_path(repo_name)
    if not repo_path.exists():
        logging.error(f"Repository '{repo_name}' not found.")
        raise HTTPException(status_code=404, detail="Repository/directory not found")

    concatenated_content = ""
    for root, dirs, files in os.walk(repo_path):
        if ".git" in dirs:
            dirs.remove(".git")
        if "node_modules" in dirs:
            dirs.remove("node_modules")
        if "__pycache__" in dirs:
            dirs.remove("__pycache__")
        for filename in files:
            file_path = Path(root) / filename
            if file_path.suffix.lower() in IGNORED_EXTENSIONS:
                continue
            rel_path = file_path.relative_to(repo_path)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                concatenated_content += f"\n--- {rel_path} ---\n{content}\n"
                logging.debug(f"Appended file {rel_path} to dump.")
            except Exception as e:
                concatenated_content += f"\n--- {rel_path} (Unreadable) ---\nError: {e}\n"
                logging.error(f"Error reading file {rel_path}: {e}")
    return {"concatenated": concatenated_content}

@app.get("/repos/{repo_name}/dump/auto-all")
def dump_auto_all_files(repo_name: str):
    """
    Return all non-ignored files via .gitignore if present, ignoring:
    - IGNORED_EXTENSIONS
    - package-lock.json
    """
    logging.info(f"Dumping auto-all files for repository '{repo_name}'.")
    repo_path = get_repo_path(repo_name)
    if not repo_path.exists():
        logging.error(f"Repository '{repo_name}' not found.")
        raise HTTPException(status_code=404, detail="Repository/directory not found")

    gitignore_path = repo_path / ".gitignore"
    if gitignore_path.exists():
        command = ["git", "ls-files", "--others", "--cached", "--exclude-standard"]
        output = run_git_command(command, cwd=str(repo_path))
        file_list = output.splitlines()
        logging.debug("Using .gitignore to filter files.")
    else:
        file_list = []
        for root, dirs, files in os.walk(repo_path):
            if ".git" in dirs:
                dirs.remove(".git")
            for f in files:
                abs_path = Path(root) / f
                rel_path = abs_path.relative_to(repo_path)
                file_list.append(str(rel_path))
    file_list_filtered = []
    for f in file_list:
        if os.path.basename(f) == "package-lock.json":
            continue
        if Path(f).suffix.lower() in IGNORED_EXTENSIONS:
            continue
        file_list_filtered.append(f)
    concatenated_content = ""
    for rel_path_str in file_list_filtered:
        file_path = repo_path / rel_path_str
        if file_path.is_file():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                concatenated_content += f"\n--- {rel_path_str} ---\n{content}\n"
                logging.debug(f"Appended file {rel_path_str} to auto-all dump.")
            except Exception as e:
                concatenated_content += f"\n--- {rel_path_str} (Unreadable) ---\nError: {e}\n"
                logging.error(f"Error reading file {rel_path_str}: {e}")
    return {"concatenated": concatenated_content}

@app.post("/repos/{repo_name}/tokens")
def calculate_tokens(repo_name: str, request: FileDumpRequest):
    logging.info(f"Calculating tokens for repository '{repo_name}'.")
    repo_path = get_repo_path(repo_name)
    if not repo_path.exists():
        logging.error(f"Repository '{repo_name}' not found.")
        raise HTTPException(status_code=404, detail="Repository/directory not found")
    total_tokens = count_tokens_in_files(repo_path, request.files)
    logging.debug(f"Total tokens for repository '{repo_name}': {total_tokens}")
    return {"total_tokens": total_tokens}

@app.get("/repos/{repo_name}/non_ignored_files")
def get_non_ignored_files(repo_name: str):
    """
    Return a flat list of all files in the repo/directory that are not* excluded by .gitignore.
    If there's no .gitignore, return all files except .git folder, ignoring package-lock.json as well.
    """
    logging.info(f"Getting non-ignored files for repository '{repo_name}'.")
    repo_path = get_repo_path(repo_name)
    if not repo_path.exists():
        logging.error(f"Repository '{repo_name}' not found.")
        raise HTTPException(status_code=404, detail="Repository/directory not found")
    gitignore_path = repo_path / ".gitignore"
    if gitignore_path.exists():
        command = ["git", "ls-files", "--others", "--cached", "--exclude-standard"]
        output = run_git_command(command, cwd=str(repo_path))
        file_list = output.splitlines()
        logging.debug("Using .gitignore for non-ignored files.")
    else:
        file_list = []
        for root, dirs, files in os.walk(repo_path):
            if ".git" in dirs:
                dirs.remove(".git")
            for f in files:
                abs_path = Path(root) / f
                rel_path = abs_path.relative_to(repo_path)
                file_list.append(str(rel_path))
    file_list = [f for f in file_list if os.path.basename(f) != "package-lock.json"]
    logging.debug(f"Non-ignored files for repo '{repo_name}': {file_list}")
    return {"files": file_list}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "32546"))
    logging.info(f"Starting server on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)