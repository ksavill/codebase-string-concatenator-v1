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

# Define directory names to show as empty in the file tree to avoid excessive loading times.
# Their content will not be traversed.
IGNORED_DIRNAMES_TO_SHOW_EMPTY_IN_TREE = {"node_modules", ".git", "__pycache__", "venv", ".venv", "env"}


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
    Directories in IGNORED_DIRNAMES_TO_SHOW_EMPTY_IN_TREE will be listed but not traversed.
    """
    if visited is None:
        visited = set()
    
    try:
        real_path = root.resolve()
    except Exception as e:
        logging.warning(f"Error resolving path {root}: {e}. Treating as empty node.")
        return {"name": root.name, "children": [], "error": f"Error resolving path: {e}"}

    if real_path in visited:
        logging.warning(f"Cycle detected: {real_path} already visited. Skipping directory: {root}")
        return {"name": root.name, "children": []} # Use root.name for the display name
    visited.add(real_path)

    # If the resolved name of the current directory is in the ignore list,
    # show it as an empty node. This handles cases where a repo itself
    # (e.g. symlinked) is one of these ignored directory types.
    if real_path.name in IGNORED_DIRNAMES_TO_SHOW_EMPTY_IN_TREE:
        logging.debug(f"Directory '{root}' (resolved to '{real_path.name}') is in IGNORED_DIRNAMES_TO_SHOW_EMPTY_IN_TREE. Showing as empty.")
        return {"name": root.name, "children": []} # Use root.name for the display name

    # If the directory is named 'repos_reference' and it's not the root itself, ignore it.
    # This check should happen for the top-level `root` argument given to the function.
    # For subdirectories named 'repos_reference', they will be skipped during iteration below.
    if root.name == "repos_reference" and root != REFERENCE_DIR:
        logging.debug(f"Ignoring directory '{root}' as it is hardcoded to be skipped (top-level).")
        return {"name": root.name, "children": []}

    logging.debug(f"Processing directory: {root} (real: {real_path})")
    tree = {"name": root.name, "children": []}
    try:
        for entry in sorted(root.iterdir(), key=lambda x: (x.is_file(), x.name)):
            if entry.is_dir():
                # Completely skip listing 'repos_reference' as a subdirectory if it's not the main one.
                if entry.name == "repos_reference" and entry.resolve() != REFERENCE_DIR.resolve():
                    logging.debug(f"Skipping subdirectory '{entry.name}' as it is 'repos_reference' and will not be listed.")
                    continue

                # For other ignored directory names, add them as empty nodes in the tree.
                if entry.name in IGNORED_DIRNAMES_TO_SHOW_EMPTY_IN_TREE:
                    logging.debug(f"Subdirectory '{entry.name}' is in IGNORED_DIRNAMES_TO_SHOW_EMPTY_IN_TREE. Adding as an empty node.")
                    tree["children"].append({"name": entry.name, "children": []})
                    continue
                
                # Regular directory: recurse
                tree["children"].append(get_file_tree(entry, visited))
            else:
                logging.debug(f"Found file: {entry}")
                tree["children"].append({"name": entry.name})
        return tree
    except Exception as e:
        logging.error(f"Error processing directory {root}: {e}")
        # In case of error reading a directory, return it as an empty node
        # to prevent breaking the whole tree structure.
        return {"name": root.name, "children": [], "error": f"Error reading directory contents: {e}"}


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
    index_path = Path(__file__).parent / "index.html"
    logging.debug(f"Serving {index_path}")
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
            # Check if the local path is inside REFERENCE_DIR to prevent recursive symlinks
            if local_path.is_relative_to(REFERENCE_DIR.resolve()):
                logging.error(f"Cannot add '{local_path}' as it is inside the reference directory '{REFERENCE_DIR}'.")
                raise HTTPException(status_code=400, detail="Cannot add a path from within the reference directory.")

            os.symlink(local_path, repo_path, target_is_directory=True)
            method_used = "symlink"
            logging.info(f"Local directory '{config.name}' added via symlink.")
        except OSError as e:
            logging.warning(f"Symlink failed for '{config.name}': {e}. Attempting copy.")
            try:
                shutil.copytree(local_path, repo_path)
                method_used = "copy"
                logging.info(f"Local directory '{config.name}' added via copy.")
            except Exception as e_copy:
                logging.error(f"Error copying local directory '{config.name}': {e_copy}")
                raise HTTPException(status_code=500, detail=str(e_copy))
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
        if p.name.startswith("."): # Skip hidden files/folders like .repo_metadata.json
            continue
        
        repo_info = {"name": p.name}
        repo_meta = metadata.get(p.name)

        if repo_meta:
            repo_info["repo_type"] = repo_meta.get("repo_type", "unknown")
            repo_info["branch"] = repo_meta.get("branch", None if repo_info["repo_type"] == "local" else "default")
            if repo_info["repo_type"] == "local":
                 repo_info["original_path"] = repo_meta.get("url")
                 repo_info["method"] = repo_meta.get("method")
            elif repo_info["repo_type"] == "git":
                 repo_info["url"] = repo_meta.get("url")
        else: # Fallback if metadata is missing (e.g. manually added folder)
            if (p / ".git").is_dir():
                repo_info["repo_type"] = "git"
                # Attempt to get current branch if possible, otherwise default
                try:
                    current_branch = run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(p))
                    repo_info["branch"] = current_branch
                except:
                    repo_info["branch"] = "default"
            else:
                repo_info["repo_type"] = "local"
        
        repos.append(repo_info)
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
        method = repo_meta.get("method")
        if method == "symlink":
            target = os.readlink(repo_path) if repo_path.is_symlink() else "N/A (not a symlink)"
            logging.info(f"Local repository '{repo_name}' is a symlink to '{target}', no refresh needed.")
            return {
                "message": f"Local directory '{repo_name}' is a symlink to '{target}'. No refresh needed."
            }
        elif method == "copy":
            source = repo_meta.get("url")
            if not source:
                logging.error("Metadata for local copied repository is missing source path.")
                raise HTTPException(status_code=500, detail="Metadata for local copied repository is missing source path.")
            source_path = Path(source).expanduser().resolve()
            if not source_path.is_dir():
                logging.error(f"Source local directory '{source}' does not exist.")
                raise HTTPException(status_code=400, detail=f"Source local directory '{source}' does not exist.")
            try:
                logging.debug(f"Refreshing copied local directory '{repo_name}' from source '{source}'.")
                shutil.rmtree(repo_path)
                shutil.copytree(source_path, repo_path)
                logging.info(f"Local directory '{repo_name}' (copy) refreshed successfully.")
                return {"message": f"Local directory '{repo_name}' (copy) refreshed successfully from source: {source}."}
            except Exception as e:
                logging.error(f"Error refreshing copied local repository '{repo_name}': {e}")
                raise HTTPException(status_code=500, detail=f"Error refreshing copied local repository: {e}")
        else: # Unknown local repo method or missing metadata
            logging.warning(f"Local repository '{repo_name}' has unknown method or missing metadata. No refresh performed.")
            return {"message": f"Local repository '{repo_name}' has unknown method. No refresh performed."}
    else: # Assumed Git repository
        git_folder = repo_path / ".git"
        if not git_folder.exists():
            logging.warning(f"No .git folder found in '{repo_name}'; assuming local directory. No refresh performed.")
            return {
                "message": f"No .git folder found in '{repo_name}'; assuming local directory. No refresh performed."
            }
        try:
            output = run_git_command(["git", "pull", "--ff-only"], cwd=str(repo_path))
            logging.info(f"Repository '{repo_name}' pulled successfully.")
            return {
                "message": f"Repository '{repo_name}' pulled successfully.",
                "details": output
            }
        except HTTPException as e: # Git command failed
             if "Not a git repository" in e.detail or "fatal: not a git repository" in e.detail :
                logging.warning(f"'{repo_name}' is not a git repository. No pull performed.")
                return {"message": f"'{repo_name}' is not a git repository. No pull performed."}
             raise # Re-raise other git errors
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
        # Pass REFERENCE_DIR to get_file_tree to correctly handle the 'repos_reference' exclusion logic
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
    for rel_path_str in request.files:
        # Sanitize rel_path_str to prevent directory traversal issues
        # Ensure it's a relative path and doesn't contain '..'
        if ".." in rel_path_str or Path(rel_path_str).is_absolute():
            logging.warning(f"Invalid relative path requested: {rel_path_str}. Skipping.")
            continue

        file_path = (repo_path / rel_path_str).resolve()

        # Double check that the resolved path is still within the repo_path
        try:
            file_path.relative_to(repo_path.resolve())
        except ValueError:
            logging.warning(f"Path traversal attempt detected or file outside repo: {rel_path_str}. Skipping.")
            continue
            
        if file_path.suffix.lower() in IGNORED_EXTENSIONS:
            logging.debug(f"Skipping ignored file: {rel_path_str}")
            continue
        if not file_path.exists() or not file_path.is_file():
            logging.error(f"File not found: {rel_path_str}")
            # Continue to next file instead of raising HTTPException to allow partial dumps
            concatenated_content += f"\n--- {rel_path_str} (File Not Found) ---\n"
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            concatenated_content += f"\n--- {rel_path_str} ---\n{content}\n"
            logging.debug(f"Appended content from file: {rel_path_str}")
        except UnicodeDecodeError:
            logging.warning(f"Skipping non-UTF-8 file: {rel_path_str}")
            concatenated_content += f"\n--- {rel_path_str} (Skipped: Not UTF-8) ---\n"
        except Exception as e:
            logging.error(f"Error reading file {rel_path_str}: {e}")
            concatenated_content += f"\n--- {rel_path_str} (Unreadable) ---\nError: {e}\n"
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
    ignored_dirs_for_walk = {".git", "node_modules", "__pycache__", "venv", ".venv", "env"}

    for root, dirs, files in os.walk(repo_path, topdown=True):
        # Modify dirs in-place to prevent os.walk from recursing into them
        dirs[:] = [d for d in dirs if d not in ignored_dirs_for_walk]
        
        for filename in files:
            file_path = Path(root) / filename
            if file_path.suffix.lower() in IGNORED_EXTENSIONS:
                continue
            
            # Ensure the file is not part of an explicitly ignored directory that might have been missed
            # if it was symlinked or named slightly differently at a deeper level.
            # This is a bit redundant if topdown=True and dirs[:] modification works as expected,
            # but provides an extra layer of safety.
            if any(ignored_dir_part in file_path.parts for ignored_dir_part in ignored_dirs_for_walk):
                # Check if the specific parent directory was on the ignore list
                # This check is mainly for files directly under an ignored dir that might not be caught by dirs[:]
                is_in_ignored_dir = False
                current_check_path = file_path.parent
                while current_check_path != repo_path and current_check_path != current_check_path.parent : # loop guard
                    if current_check_path.name in ignored_dirs_for_walk:
                        is_in_ignored_dir = True
                        break
                    current_check_path = current_check_path.parent
                if is_in_ignored_dir:
                    logging.debug(f"Skipping file {file_path} as it is in an ignored directory.")
                    continue

            rel_path = file_path.relative_to(repo_path)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                concatenated_content += f"\n--- {rel_path} ---\n{content}\n"
                logging.debug(f"Appended file {rel_path} to dump.")
            except UnicodeDecodeError:
                concatenated_content += f"\n--- {rel_path} (Skipped: Not UTF-8) ---\n"
                logging.warning(f"Skipping non-UTF-8 file {rel_path} in dump/all.")
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

    file_list_to_dump = get_non_ignored_files_list(repo_path, repo_name) # Use helper

    concatenated_content = ""
    for rel_path_str in file_list_to_dump:
        # Path sanitization and validation already handled by get_non_ignored_files_list
        # and the subsequent file reading logic.
        file_path = repo_path / rel_path_str
        
        # This check is slightly redundant as get_non_ignored_files_list should only return files
        # but good for safety.
        if not file_path.is_file():
            logging.warning(f"Path '{rel_path_str}' from git ls-files is not a file. Skipping.")
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            concatenated_content += f"\n--- {rel_path_str} ---\n{content}\n"
            logging.debug(f"Appended file {rel_path_str} to auto-all dump.")
        except UnicodeDecodeError:
            concatenated_content += f"\n--- {rel_path_str} (Skipped: Not UTF-8) ---\n"
            logging.warning(f"Skipping non-UTF-8 file {rel_path_str} in dump/auto-all.")
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
    
    valid_files = []
    for rel_path_str in request.files:
        if ".." in rel_path_str or Path(rel_path_str).is_absolute():
            logging.warning(f"Invalid relative path for token count: {rel_path_str}. Skipping.")
            continue
        
        file_path_check = (repo_path / rel_path_str).resolve()
        try:
            file_path_check.relative_to(repo_path.resolve())
            valid_files.append(rel_path_str)
        except ValueError:
            logging.warning(f"Path traversal attempt for token count: {rel_path_str}. Skipping.")
            continue

    total_tokens = count_tokens_in_files(repo_path, valid_files)
    logging.debug(f"Total tokens for repository '{repo_name}': {total_tokens}")
    return {"total_tokens": total_tokens}


def get_non_ignored_files_list(repo_path: Path, repo_name: str) -> List[str]:
    """Helper function to get a list of non-ignored files."""
    gitignore_path = repo_path / ".gitignore"
    tracked_files = []

    if gitignore_path.exists() and (repo_path / ".git").is_dir():
        try:
            # --cached: All files tracked by git
            # --others: All untracked files (not in .gitignore)
            # --exclude-standard: Use .gitignore, .git/info/exclude, global exclude
            command = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
            output = run_git_command(command, cwd=str(repo_path))
            tracked_files = output.splitlines()
            logging.debug(f"Using .gitignore for non-ignored files in {repo_name}.")
        except HTTPException as e:
            logging.warning(f"git ls-files failed for {repo_name} (detail: {e.detail}), falling back to os.walk.")
            tracked_files = [] # Fallback will execute
    
    if not tracked_files: # Fallback if no .git, no .gitignore, or git ls-files failed
        logging.debug(f"No .gitignore or .git found, or git ls-files failed for {repo_name}. Using os.walk to list files.")
        ignored_dirs_for_walk = {".git", "node_modules", "__pycache__", "venv", ".venv", "env"}
        for root, dirs, files in os.walk(repo_path, topdown=True):
            dirs[:] = [d for d in dirs if d not in ignored_dirs_for_walk]
            for f_name in files:
                abs_path = Path(root) / f_name
                # Basic check to ensure we are within repo_path, though os.walk should handle this.
                try:
                    rel_path = abs_path.relative_to(repo_path)
                    tracked_files.append(str(rel_path))
                except ValueError:
                    logging.warning(f"File {abs_path} is outside of repo path {repo_path}. Skipping.")

    # Filter common lock files and ignored extensions from the list
    final_file_list = []
    for f_path_str in tracked_files:
        f_path = Path(f_path_str) # Create Path object for suffix and name checks
        if f_path.name == "package-lock.json" or f_path.name == "yarn.lock" or f_path.name == "pnpm-lock.yaml":
            logging.debug(f"Excluding lock file: {f_path_str}")
            continue
        if f_path.suffix.lower() in IGNORED_EXTENSIONS:
            logging.debug(f"Excluding by extension: {f_path_str}")
            continue
        
        # Ensure the file actually exists and is a file (git ls-files can list deleted files if not careful)
        # and is within the repo directory (security for paths from git ls-files)
        full_file_path = (repo_path / f_path_str).resolve()
        if not full_file_path.is_file():
            logging.debug(f"File {f_path_str} listed by git ls-files (or walk) does not exist or is not a file. Skipping.")
            continue
        try:
            full_file_path.relative_to(repo_path.resolve())
        except ValueError:
            logging.warning(f"Path traversal attempt or file outside repo from git ls-files: {f_path_str}. Skipping.")
            continue

        final_file_list.append(f_path_str)
        
    return final_file_list

@app.get("/repos/{repo_name}/non_ignored_files")
def get_non_ignored_files_endpoint(repo_name: str):
    """
    Return a flat list of all files in the repo/directory that are not* excluded by .gitignore.
    If there's no .gitignore, return all files except common ignored dirs and files.
    Also filters IGNORED_EXTENSIONS and common lock files.
    """
    logging.info(f"Getting non-ignored files for repository '{repo_name}'.")
    repo_path = get_repo_path(repo_name)
    if not repo_path.exists():
        logging.error(f"Repository '{repo_name}' not found.")
        raise HTTPException(status_code=404, detail="Repository/directory not found")

    file_list = get_non_ignored_files_list(repo_path, repo_name)
    
    logging.debug(f"Non-ignored files for repo '{repo_name}': {file_list}")
    return {"files": file_list}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "32546"))
    logging.info(f"Starting server on port {port}")
    # Determine script's directory to correctly locate index.html
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir) # Change CWD so index.html can be found by FileResponse relative to CWD if not absolute
    
    # The FileResponse in index() endpoint now uses Path(__file__).parent / "index.html"
    # which is absolute, so chdir might not be strictly necessary for that specific endpoint,
    # but good practice if other relative paths are used.

    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True, reload_dirs=[str(script_dir)])