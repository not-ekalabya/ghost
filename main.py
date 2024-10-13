import os
import json
import streamlit as st
from anthropic import AnthropicVertex
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple
from functools import lru_cache
import pickle
import logging
from pathlib import Path
import subprocess
import asyncio
import git

# todo
# Fix changes not being implemented in directory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextEmbedder:
    def __init__(self, model_name="BAAI/bge-large-en-v1.5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    @lru_cache(maxsize=1000)
    def embed_single(self, text: str) -> torch.Tensor:
        encoded_input = self.tokenizer(
            [text], padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embedding = model_output[0][:, 0]
        return F.normalize(embedding, p=2, dim=1)

    def embed(self, texts: List[str]) -> torch.Tensor:
        return torch.cat([self.embed_single(text) for text in texts])


def process_codebase(
    directory: str, cache_file: str = "codebase_cache.pkl"
) -> List[Tuple[str, str]]:
    if os.path.exists(cache_file):
        logger.info("Loading codebase from cache...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    code_snippets = []
    ignored_dirs = {"node_modules", "__pycache__", ".git", "venv", "env"}
    allowed_extensions = {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".cs",
        ".go",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".rs",
        ".json",
        ".md",
    }

    logger.info(f"Scanning directory: {directory}")
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        for file in files:
            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file)
            if ext.lower() in allowed_extensions:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    code_snippets.append((file_path, content))
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {e}")

    logger.info(f"Total code snippets found: {len(code_snippets)}")

    # Cache the processed codebase
    with open(cache_file, "wb") as f:
        pickle.dump(code_snippets, f)

    return code_snippets


class CodebaseEmbeddings:
    def __init__(self, code_snippets: List[Tuple[str, str]], embedder: TextEmbedder):
        self.code_snippets = code_snippets
        self.embedder = embedder
        self.embeddings = None

    def compute_embeddings(self):
        if self.embeddings is None:
            logger.info("Computing code embeddings...")
            self.embeddings = self.embedder.embed(
                [snippet[1] for snippet in self.code_snippets]
            )
        return self.embeddings


def retrieve_relevant_code(
    query: str,
    codebase_embeddings: CodebaseEmbeddings,
    embedder: TextEmbedder,
    top_k: int = 3,
) -> List[Tuple[str, str, float]]:
    if not codebase_embeddings.code_snippets:
        logger.warning("No code snippets found in the codebase.")
        return []

    query_with_instruction = (
        f"Represent this sentence for retrieving relevant code snippets: {query}"
    )
    query_embedding = embedder.embed([query_with_instruction])

    code_embeddings = codebase_embeddings.compute_embeddings()

    scores = (query_embedding @ code_embeddings.T).squeeze(0)
    top_k = min(top_k, len(codebase_embeddings.code_snippets))
    top_indices = torch.argsort(scores, descending=True)[:top_k]

    return [
        (
            codebase_embeddings.code_snippets[i][0],
            codebase_embeddings.code_snippets[i][1],
            scores[i].item(),
        )
        for i in top_indices
    ]


class ClaudeClient:
    def __init__(self, project_id: str, region: str):
        self.client = AnthropicVertex(project_id=project_id, region=region)

    def generate_response(self, messages: List[dict]) -> str:
        response = self.client.messages.create(
            model="claude-3-5-sonnet@20240620",
            system=system_prompt,
            max_tokens=8192,
            messages=messages,
        )
        return response.content[0].text


class RAGApplication:
    def __init__(self, codebase_directory: str, project_id: str, region: str):
        self.embedder = TextEmbedder()
        self.code_snippets = process_codebase(codebase_directory)
        self.codebase_embeddings = CodebaseEmbeddings(self.code_snippets, self.embedder)
        self.claude_client = ClaudeClient(project_id, region)
        self.codebase_directory = codebase_directory

    def answer_question(self, question: str) -> str:
        relevant_code = retrieve_relevant_code(
            question, self.codebase_embeddings, self.embedder
        )

        context = "Here are some relevant code snippets from the codebase:\n\n"
        for file_path, code, score in relevant_code:
            context += f"File: {file_path}\nRelevance Score: {score:.2f}\n\n```\n{code}\n```\n\n"

        messages = [
            {
                "role": "user",
                "content": f"{question}\nHere is some context about the project\nContext:\n{context}\n\nOUTPUT VALID JSON",
            },
        ]

        return self.claude_client.generate_response(messages)

    def implement_changes(self, changes: List[dict]) -> List[dict]:
        results = []
        for change in changes:
            action = change["action"]

            if action in ["create", "modify"]:
                path = os.path.join(self.codebase_directory, change["path"])
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    f.write(change["content"])
                results.append({"action": action, "path": path, "status": "success"})

            elif action == "delete":
                path = os.path.join(self.codebase_directory, change["path"])
                if os.path.exists(path):
                    os.remove(path)
                    results.append(
                        {"action": action, "path": path, "status": "success"}
                    )
                else:
                    results.append(
                        {
                            "action": action,
                            "path": path,
                            "status": "error",
                            "message": "File not found",
                        }
                    )

            elif action == "execute":
                command = change["command"]
                try:
                    result = subprocess.run(
                        command,
                        shell=True,
                        check=True,
                        capture_output=True,
                        text=True,
                        cwd=self.codebase_directory,
                    )
                    results.append(
                        {
                            "action": action,
                            "command": command,
                            "output": result.stdout,
                            "status": "success",
                        }
                    )
                except subprocess.CalledProcessError as e:
                    results.append(
                        {
                            "action": action,
                            "command": command,
                            "output": e.stderr,
                            "status": "error",
                        }
                    )

        return results


system_prompt = """
The assistant is an expert software engineer called ghost. It analyzes the given context, identifying key aspects of the project important for software engineering such as the technologies used, the purpose of the project, and how they align with the query. Ghost outputs functional and clean code which resonates with the technologies used. Ghost provides comments to write self-explaining code that can be implemented straight to the project.

Ghost can now execute shell commands in its directory. When a shell command needs to be executed, Ghost will include it in the JSON response as part of the changes.

Ghost outputs its response in the form of a JSON object with the following structure:

{
    "description": "A brief description of the changes",
    "makrkdown": "all the code explanation"
    "changes": [
        {
            "action": "create|modify|delete|execute",
            "path": "path/to/file",
            "content": "New or modified content of the file",
            "command": "Shell command to execute"
        }
    ],
    "summary": "A summary of the changes implemented and commands executed"
}

The 'changes' array can contain multiple objects, each representing a file to be created, modified, or deleted, or a command to be executed. For 'delete' actions, the 'content' field should be omitted. For 'execute' actions, the 'command' field should contain the shell command to be executed, and 'path' and 'content' fields should be omitted.

GHOST OUTPUTS ONLY IN VALID JSON FORMAT. THE OUTPUT CAN BE READILY PARSED AS A JSON OBEJCT.
Any other information is except the json is not shown.
"""


def implement_changes(changes):
    results = []
    for change in changes:
        action = change["action"]

        if action == "create" or action == "modify":
            path = change["path"]
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(change["content"])
            results.append({"action": action, "path": path, "status": "success"})

        elif action == "delete":
            path = change["path"]
            if os.path.exists(path):
                os.remove(path)
                results.append({"action": action, "path": path, "status": "success"})
            else:
                results.append(
                    {
                        "action": action,
                        "path": path,
                        "status": "error",
                        "message": "File not found",
                    }
                )

        elif action == "execute":
            command = change["command"]
            try:
                result = subprocess.run(
                    command, shell=True, check=True, capture_output=True, text=True
                )
                results.append(
                    {
                        "action": action,
                        "command": command,
                        "output": result.stdout,
                        "status": "success",
                    }
                )
            except subprocess.CalledProcessError as e:
                results.append(
                    {
                        "action": action,
                        "command": command,
                        "output": e.stderr,
                        "status": "error",
                    }
                )

    return results


@st.cache_resource
def initialize_rag_application(repo_or_dir, *, is_local=False, local_dir="repo"):
    if is_local:
        repo_path = Path(repo_or_dir)
    else:
        github_repo_url = f"https://github.com/{repo_or_dir}.git"
        repo_path = Path(local_dir) / repo_or_dir.replace("/", "_")

        async def clone_repo():
            if not repo_path.exists():
                logger.info(f"Cloning repository from {github_repo_url}...")
                git.Repo.clone_from(github_repo_url, str(repo_path))
            else:
                logger.info(
                    f"Repository already exists at {repo_path}. Skipping clone."
                )

        asyncio.run(clone_repo())

    project_id = "GCP_PROJECT"
    region = "GCP_LOCATION"

    return RAGApplication(str(repo_path), project_id, region)


# Streamlit UI
st.title("Code Assistant RAG")

# User input
dir = st.text_input("Project Directory:")
dir_type = st.selectbox("Type of repository:", ["github", "local"])
query = st.text_area("Enter your coding question:")

rag_app = initialize_rag_application(dir, is_local=dir_type == "local")

if rag_app is None:
    st.error(
        "Failed to initialize RAG application. Please check your codebase directory and try again."
    )
else:
    if st.button("Get Answer"):
        if query:
            with st.spinner("Generating answer..."):
                try:
                    answer = rag_app.answer_question(query)
                    st.write("Answer:", answer)

                    try:
                        response_json = json.loads(answer)
                        if (
                            isinstance(response_json, dict)
                            and "changes" in response_json
                        ):
                            st.write("Proposed changes:")
                            st.json(response_json)

                            if st.button("Implement Changes"):
                                results = rag_app.implement_changes(
                                    response_json["changes"]
                                )
                                st.write("Implementation results:")
                                st.json(results)
                                st.success("Changes implemented successfully!")
                        else:
                            st.write("No changes to implement.")
                    except json.JSONDecodeError:
                        st.write(
                            "The response is not in JSON format. No changes to implement."
                        )
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a question.")
