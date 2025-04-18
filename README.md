# Repo File Concatenator for LLM Context

This application allows you to quickly concatenate selected files from either local directories or Git repositories into a single string. It's designed to streamline the process of providing extensive code context to large language models (LLMs), saving significant manual effort.

## Features

- **Repository Management**: Supports adding Git repositories (with automatic cloning and pulling) or local directories.
- **File Selection**: Interactive file tree for selecting individual files to include.
- **Auto-Select Non-Ignored Files**: Quickly select files not excluded by `.gitignore`.
- **Token Calculation**: Estimates the total token count of selected files, essential for managing context size with LLMs.
- **Concatenation Output**: Generates an easily copyable concatenated text output from selected files.
- **Dockerized Setup**: Simple containerized deployment using Docker and Docker Compose.

## Installation

Ensure Docker and Docker Compose are installed on your system.

Clone this repository:

```bash
git clone <your-repo-url>
cd <repo-folder>
```

Build and run the Docker container:

```bash
docker-compose up --build
```

Access the application at: `http://localhost:32546`

## Usage

### Managing Repositories

- Add repositories or directories using the interface by specifying a name and either a Git URL or a local path.
- Pull updates from Git repositories with one click.
- Easily remove unwanted repositories/directories.

### Selecting Files

- Load the file tree from any added repository or directory.
- Manually select individual files or auto-select all non-ignored files.

### Generating Output

- Concatenate the contents of selected files into a single string for easy copying.
- Optionally calculate the total tokens of selected content to manage your LLM inputs effectively.

## Technologies Used

- **FastAPI**: Lightweight Python backend API.
- **Uvicorn**: ASGI server for efficient performance.
- **Tiktoken**: Accurate token counting compatible with OpenAI's tokenizer.
- **Docker**: Containerization for ease of deployment and consistency across environments.

## Project Structure

```
.
├── DockerFile.dockerfile
├── docker-compose.yml
├── index.html
├── main.py
├── requirements.txt
└── repos_reference
```

- `repos_reference`: Directory containing cloned repositories or linked local directories.

## Contributions

Feel free to fork, improve, or contribute enhancements through pull requests.
