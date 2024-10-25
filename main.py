import os
import json
import streamlit as st
from anthropic import (
    AnthropicVertex,
    APIStatusError,
    APITimeoutError,
    APIConnectionError,
)
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
import difflib

import tempfile
import ast
from dataclasses import dataclass
from typing import List, Dict, Any
import traceback
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SETUP GCP PROJECT AT 784
# SETUP GCP PROJECT AT 784
# SETUP GCP PROJECT AT 784


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
    # if os.path.exists(cache_file):
    #     logger.info("Loading codebase from cache...")
    #     with open(cache_file, "rb") as f:
    #         return pickle.load(f)

    code_snippets = []
    ignored_dirs = {"node_modules", "__pycache__", ".git", "venv", "env"}
    ignored_files = {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        ".gitignore",
        ".env",
    }
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

    # First check for README files
    readme_paths = [
        os.path.join(directory, f) for f in ["README.md", "Readme.md", "readme.md"]
    ]
    for readme_path in readme_paths:
        if os.path.exists(readme_path):
            try:
                with open(readme_path, "r", encoding="utf-8") as f:
                    content = f.read()
                code_snippets.append((readme_path, content))
                logger.info(f"Added README file: {readme_path}")
                break
            except Exception as e:
                logger.error(f"Error reading README file {readme_path}: {e}")

    logger.info(f"Scanning directory: {directory}")
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        for file in files:
            if file in ignored_files:
                continue
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
        self.cache_file = "codebase_cache.pkl"

    def compute_embeddings(self):
        if self.embeddings is None:
            logger.info("Computing code embeddings...")
            self.embeddings = self.embedder.embed(
                [snippet[1] for snippet in self.code_snippets]
            )
        return self.embeddings

    def update_embeddings(self, file_path: str, content: str):
        for i, (path, _) in enumerate(self.code_snippets):
            if path == file_path:
                self.code_snippets[i] = (file_path, content)
                break
        else:
            self.code_snippets.append((file_path, content))

        self.embeddings = None  # Reset embeddings to force recomputation
        self.save_cache()

    def delete_embedding(self, file_path: str):
        self.code_snippets = [
            snippet for snippet in self.code_snippets if snippet[0] != file_path
        ]
        self.embeddings = None  # Reset embeddings to force recomputation
        self.save_cache()

    def save_cache(self):
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.code_snippets, f)

    def reload_database(self, directory: str):
        self.code_snippets = process_codebase(directory, cache_file=self.cache_file)
        self.embeddings = None
        logger.info("Vector database reloaded.")


def retrieve_relevant_code(
    query: str,
    codebase_embeddings: CodebaseEmbeddings,
    embedder: TextEmbedder,
    top_k: int = 3,
) -> List[Tuple[str, str, float]]:
    if not codebase_embeddings.code_snippets:
        logger.warning("No code snippets found in the codebase.")
        return []

    query_with_instruction = f"'{query}' - This is a prompt given to a software engineer. Represent all the reference code the engineer needs. Always mention the name of the file and the class and function ( if any ) with the respective code snippets"
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
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet@20240620",
                system=system_prompt,
                max_tokens=8192,
                messages=messages,
            )

            # Try to parse the response as JSON
            # print(response.content[0].text)
            logger.info(f"claude client: {response.content[0].text}")
            return response.content[0].text

        except (APIStatusError, APITimeoutError, APIConnectionError) as e:
            if "context length exceeded" in str(e).lower():
                raise
            else:
                logger.error(f"API error: {str(e)}")
                return json.dumps(
                    {
                        "error": "API error",
                        "description": str(e),
                        "changes": [],
                        "summary": "No changes implemented due to API error",
                    }
                )


class ChatMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "ChatMessage":
        return cls(role=data["role"], content=data["content"])


def extract_json(response):
    # Initialize variables to track braces and JSON candidates
    stack = []
    json_candidates = []
    current_candidate = ""

    # Iterate through each character in the response
    for char in response:
        if char == "{":
            # Start a new JSON candidate if stack is empty
            if not stack:
                current_candidate = char
            else:
                current_candidate += char
            stack.append(char)
        elif char == "}":
            if stack:
                current_candidate += char
                stack.pop()
                # If stack is empty, we found a complete JSON candidate
                if not stack:
                    json_candidates.append(current_candidate)
                    current_candidate = ""
        else:
            # Add characters to current candidate if within braces
            if stack:
                current_candidate += char

    # If no JSON-like content is found, return None
    if not json_candidates:
        return None

    # Try to parse each potential JSON string
    for json_str in json_candidates:
        try:
            # Attempt to parse the JSON
            json_obj = json.loads(json_str)
            return json_obj
        except json.JSONDecodeError:
            # If parsing fails, clean up and try again
            cleaned_json_str = re.sub(r"([{,]\s*)(\w+)(\s*:)", r'\1"\2"\3', json_str)
            try:
                json_obj = json.loads(cleaned_json_str, strict=False)
                return json_obj
            except json.JSONDecodeError:
                continue  # Move to the next candidate if parsing fails

    # If no valid JSON is found, return None
    return None


class RAGApplication:

    def __init__(self, codebase_directory: str, project_id: str, region: str):
        self.embedder = TextEmbedder()
        self.code_snippets = process_codebase(codebase_directory)
        self.codebase_embeddings = CodebaseEmbeddings(self.code_snippets, self.embedder)
        self.claude_client = ClaudeClient(project_id, region)
        self.codebase_directory = codebase_directory
        # Initialize with empty chat history
        self.chat_history = []
        logger.info(
            f"Initialized RAGApplication with codebase directory: {self.codebase_directory}"
        )

    def answer_question(
        self, question: str, images: List[Tuple[str, bytes]] = None, attempts=3
    ) -> str:
        max_attempts = attempts
        attempt = 0
        last_response_json = None

        # Convert chat history to format expected by Claude
        formatted_messages = [
            {"role": msg["role"], "content": [{"type": "text", "text": msg["content"]}]}
            for msg in self.chat_history
        ]

        # Add current question with a note about context availability
        question_with_note = f"{question}\n\nNote: If you need context about the     codebase, you can request it by responding with a JSON that includes     'needs_context': true"
        formatted_messages.append(
            {"role": "user", "content": [{"type": "text", "text": question_with_note}]}
        )

        while attempt < max_attempts:
            try:
                response = self.claude_client.generate_response(formatted_messages)
                response_json = extract_json(response)

                if response_json and response_json.get("needs_context"):
                    # Model requested context, so let's provide it
                    embedder = TextEmbedder()
                    codebase_embeddings = CodebaseEmbeddings(
                        self.code_snippets, embedder
                    )
                    relevant = retrieve_relevant_code(
                        query=question,
                        codebase_embeddings=codebase_embeddings,
                        embedder=embedder,
                    )

                    context_message = f"Here is the relevant context about the     project:\n{relevant}\n\nNote: The relevant context is provided     in form of a list of tuples. Each tuple has 3 values in this     specific order - (<path to the file>, <content of the file>,     <relevance of the file in the range 0-1>)"
                    formatted_messages.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": context_message}],
                        }
                    )
                    continue  # Retry with context

                if response_json:
                    # Update chat history with the successful exchange
                    self.chat_history.append({"role": "user", "content": question})
                    self.chat_history.append({"role": "assistant", "content": response})
                    return json.dumps(response_json)

                attempt += 1
                last_response_json = response_json

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                attempt += 1
                continue

        # If we've exhausted attempts, return the last response with error info
        final_response = last_response_json or {
            "description": "Failed to process request",
            "changes": [],
            "summary": "Error occurred during processing",
        }

        return json.dumps(final_response)

    def _truncate_chat_history(self, messages: List[dict]) -> List[dict]:
        """Truncate chat history to reduce context length."""
        if len(messages) <= 2:
            return messages

        # Remove the oldest message that is not the system message
        return [messages[0]] + messages[2:]

    def _should_execute(self, response_json: dict) -> bool:
        """Check if the last change in the response is an execute command."""
        if "changes" in response_json and response_json["changes"]:
            last_change = response_json["changes"][-1]
            return last_change.get("action") == "execute"
        return False

    def parse_unified_diff(self, diff_text: str) -> List[Tuple[str, str, int, str]]:
        """Enhanced parse_unified_diff with better handling of complex diffs"""
        # Handle escaped newlines and quotes
        diff_text = diff_text.replace("\\n", "\n").replace('\\"', '"')
        changes = []
        current_line = 0
        current_chunk = []
        in_chunk = False

        for line in diff_text.splitlines():
            # Handle diff headers
            if line.startswith("---") or line.startswith("+++"):
                continue

            # Handle chunk headers
            if line.startswith("@@"):
                if in_chunk and current_chunk:
                    # Process previous chunk
                    changes.extend(self._process_chunk(current_chunk, current_line))
                    current_chunk = []

                try:
                    # Parse the chunk header
                    header = line.split("@@")[1].strip()
                    _, new = header.split(" ")
                    current_line = int(new.split(",")[0].lstrip("+")) - 1
                    in_chunk = True
                    continue
                except (IndexError, ValueError):
                    logger.error(f"Failed to parse diff header: {line}")
                    continue

            if in_chunk:
                current_chunk.append((line, current_line))
                if not line.startswith(("+", "-")):
                    current_line += 1

        # Process the last chunk
        if current_chunk:
            changes.extend(self._process_chunk(current_chunk, current_line))

        return changes

    def _process_chunk(
        self, chunk: List[Tuple[str, int]], start_line: int
    ) -> List[Tuple[str, str, int, str]]:
        """Process a single chunk of diff and return the changes"""
        changes = []
        current_line = start_line

        for line, _ in chunk:
            if line.startswith("+"):
                changes.append(("add", line[1:], current_line, ""))
                current_line += 1
            elif line.startswith("-"):
                changes.append(("remove", line[1:], current_line, ""))
            else:
                current_line += 1

        return changes

    def apply_diff(self, file_path: str, diff_content: str) -> str:
        """Improved apply_diff with better handling of complex changes"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            lines = []
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    lines = f.readlines()
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {str(e)}")
                return ""

        # Normalize line endings
        lines = [line.rstrip("\r\n") + "\n" for line in lines]

        # Parse and apply changes
        changes = self.parse_unified_diff(diff_content)

        # Sort changes by line number in reverse order to avoid offset issues
        changes.sort(key=lambda x: x[2], reverse=True)

        for change_type, content, line_num, _ in changes:
            try:
                if change_type == "add":
                    if line_num >= len(lines):
                        lines.append(content + "\n")
                    else:
                        lines.insert(line_num, content + "\n")
                elif change_type == "remove":
                    if 0 <= line_num < len(lines):
                        lines.pop(line_num)
            except Exception as e:
                logger.error(f"Failed to apply change at line {line_num}: {str(e)}")
                continue

        return "".join(lines)

    def generate_diff(self, file_path: str, new_content: str) -> str:
        """Generate a unified diff between existing file content and new content"""
        try:
            with open(file_path, "r") as f:
                old_content = f.read().splitlines()
        except FileNotFoundError:
            old_content = []

        new_content = new_content.splitlines()

        # Create unified diff
        diff = list(
            difflib.unified_diff(
                old_content,
                new_content,
                fromfile=file_path,
                tofile=file_path,
                lineterm="",
                n=3,  # Context lines
            )
        )

        return "\n".join(diff) if diff else ""

    def implement_changes(self, changes: List[dict]) -> List[dict]:
        """Improved implement_changes with better error handling and atomic operations"""
        results = []
        temp_files = {}  # Store temporary changes before committing

        # First pass: Parse all changes into temp_files
        for change in changes:
            if change["action"] in ["create", "modify"]:
                path = os.path.join(self.codebase_directory, change["path"])
                if "content" in change:
                    if change["action"] == "create":
                        # For create action, use content directly
                        temp_files[path] = change["content"]
                    else:
                        # For modify action, parse the unified diff content
                        lines = change["content"].splitlines()
                        file_content = []
                        current_line = 0

                        # Skip the header lines (---, +++)
                        header_count = 0
                        i = 0
                        while header_count < 2 and i < len(lines):
                            if lines[i].startswith("---") or lines[i].startswith("+++"):
                                header_count += 1
                            i += 1

                        # Process the diff hunks
                        while i < len(lines):
                            line = lines[i]
                            if line.startswith("@@"):
                                # Parse the hunk header
                                match = re.match(
                                    r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line
                                )
                                if match:
                                    current_line = int(match.group(1)) - 1
                            elif line.startswith("+"):
                                file_content.insert(current_line, line[1:])
                                current_line += 1
                            elif line.startswith("-"):
                                # Skip removed lines
                                pass
                            elif not line.startswith(
                                "\\"
                            ):  # Ignore "No newline" markers
                                file_content.insert(current_line, line)
                                current_line += 1
                            i += 1

                        temp_files[path] = "\n".join(file_content)
                else:
                    logger.error(
                        f"No content provided for {change['action']} action on {path}"
                    )
                    results.append(
                        {
                            "action": change["action"],
                            "path": path,
                            "status": "error",
                            "message": "No content provided",
                        }
                    )

        # Second pass: Validate all changes can be applied
        for path, content in temp_files.items():
            if not content and os.path.exists(path):
                logger.error(f"Failed to generate valid content for {path}")
                results.append(
                    {
                        "action": "modify",
                        "path": path,
                        "status": "error",
                        "message": "Failed to generate valid content",
                    }
                )
                return results

        # Third pass: Actually implement the changes
        for change in changes:
            action = change["action"]
            try:
                if action in ["create", "modify"]:
                    path = os.path.join(self.codebase_directory, change["path"])
                    content = temp_files[path]
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)
                    self.codebase_embeddings.update_embeddings(path, content)
                    results.append(
                        {
                            "action": action,
                            "path": path,
                            "status": "success",
                            "diff": change.get("content", ""),
                        }
                    )
                elif action == "delete":
                    path = os.path.join(self.codebase_directory, change["path"])
                    if os.path.exists(path):
                        os.remove(path)
                        self.codebase_embeddings.delete_embedding(path)
                        results.append(
                            {"action": action, "path": path, "status": "success"}
                        )
                    else:
                        results.append(
                            {
                                "action": action,
                                "path": path,
                                "status": "error",
                                "message": "File does not exist",
                            }
                        )
                elif action == "execute":
                    command = change["command"]
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
            except Exception as e:
                logger.error(f"Error implementing {action}: {str(e)}")
                results.append(
                    {
                        "action": action,
                        "path": change.get("path", ""),
                        "status": "error",
                        "message": str(e),
                    }
                )
                # Rollback any changes made so far
                for path, content in temp_files.items():
                    if os.path.exists(path):
                        try:
                            with open(path, "r", encoding="utf-8") as f:
                                original = f.read()
                            with open(path, "w", encoding="utf-8") as f:
                                f.write(original)
                        except Exception as rollback_error:
                            logger.error(
                                f"Error during rollback: {str(rollback_error)}"
                            )
                return results

        return results

    def reload_database(self):
        self.codebase_embeddings.reload_database(self.codebase_directory)


system_prompt = """
<role>
The assistant is an expert software engineer called ghost. It analyzes the given context, identifying key aspects of the project important for software engineering such as the technologies used, the purpose of the project, and how they align with the query. Ghost outputs functional and clean code which resonates with the technologies used. 

Ghost provides comments to write self-explaining code that can be implemented straight to the project.

Ghost can execute shell commands in its directory when necessary. When a shell command needs to be executed, Ghost will include it in the JSON response as part of the changes.

Ghost can also process and analyze images provided in the conversation. When images are present, Ghost will consider them in the context of the query and provide relevant insights or code modifications based on the image content.

</role>

<context>
If you need information about the codebase to better answer a question, you can request it by responding with a JSON that includes "needs_context": true. The system will then provide relevant code snippets from the project.
</context>

<format>

Ghost outputs its response in the form of a JSON object with the following structure:

{
    "description": "A brief description of the response or changes",
    "markdown": "Detailed explanation or response content",
    "changes": [ // if necessary
        {
            "action": "create|modify|delete|execute",
            "path": "path/to/file",
            "content": "New or modified content of the file in diff format",
            "command": "Shell command to execute"
        }
    ],
    "summary": "A summary of the response or changes implemented"
}

For code-related queries, when modifying files, the content should be in a unified diff format:
- Start with '---' and '+++' headers showing the file being modified
- Include @@ markers to indicate the location of changes
- Use '-' for removed lines and '+' for added lines
- Preserve context lines without markers

Example diff format:
{
    "action": "modify", 
    "path": "example.py",
    "content": ""
        --- example.py
        +++ example.py
        @@ -10,7 +10,7 @@
         def existing_function():
        -    old_code = 'remove'
        +    new_code = 'add'
             other_code()
        ""
}


</format>

<important>
Try to implement the changes in the current working directory.
For example - for a query like "Can you setup a react app in the current working directory?", run a command like "npx create-react-app ." instead of a command like "npx create-react-app <app-name>"

Leave the changes array empty for non-code related tasks such as - "Can you write a detailed background about this project?" or "What technologies does this project use?". In case of questions like these respond in the "markdown" key of the JSON object without executing changes.
</important>

<notes>
For code-related queries:
- Use the 'create', 'modify', or 'delete' actions for file operations.
- Use the 'execute' action when a command needs to be run.
- For 'modify' actions, use a diff-like format in the 'content' field.

For non-code-related queries:
- Provide an appropriate response in the 'markdown' field.
- Do not include unnecessary 'execute' actions or empty 'changes' arrays.

OUTPUT ONLY IN VALID JSON FORMAT. THE OUTPUT CAN BE READILY PARSED AS A JSON OBJECT.
</notes>

"""


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

    project_id = "YOUR-GCP-PROJECT"
    region = "europe-west1"

    return RAGApplication(str(repo_path), project_id, region)


# Streamlit UI
st.set_page_config(layout="wide")

# Initialize session state
if "rag_app" not in st.session_state:
    st.session_state.rag_app = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for project configuration
with st.sidebar:
    st.title("Project Configuration")
    dir = st.text_input("Project Directory:")
    dir_type = st.selectbox("Type of repository:", ["github", "local"])
    if st.button("Initialize Project"):
        st.session_state.rag_app = initialize_rag_application(
            dir, is_local=dir_type == "local"
        )
        if st.session_state.rag_app:
            st.success("Project initialized successfully!")
        else:
            st.error(
                "Failed to initialize project. Please check your inputs and try again."
            )

    # Add reload database button
    if st.session_state.rag_app and st.button("Reload Database"):
        st.session_state.rag_app.reload_database()
        st.success("Database reloaded successfully!")

# Main chat interface
st.title("Chat with Ghost")

# Chat history display
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# File uploader for images and other files
uploaded_files = st.file_uploader("Upload images or files", accept_multiple_files=True)
max_attempts = st.number_input("Maximum attempts: ", step=1, value=3)

# User input
if prompt := st.chat_input("What would you like to know?"):
    if st.session_state.rag_app:
        with st.chat_message("user"):
            st.write(prompt)
            for file in uploaded_files:
                st.write(f"Uploaded file: {file.name}")

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Prepare images for the RAG application
                images = [
                    (file.name, file.getvalue())
                    for file in uploaded_files
                    if file.type.startswith("image/")
                ]

                codebase = process_codebase(dir)
                embedder = TextEmbedder()

                codebase_embeddings = CodebaseEmbeddings(
                    code_snippets=codebase, embedder=embedder
                )
                relevant = retrieve_relevant_code(
                    query=prompt,
                    codebase_embeddings=codebase_embeddings,
                    embedder=embedder,
                )

                logger.info(relevant)

                response = st.session_state.rag_app.answer_question(
                    f"{prompt}\n\n Here is the relevant context you would need about the project-\n {relevant}\n\n Note: The relevant context is provided in form of a list of tuples. Each tuple has 3 values in this specific order - (<path to the file>, <content of the file>, <relevance of the file in the range 0-1>)",
                    images,
                    max_attempts,
                )

                try:
                    response_json = extract_json(response)
                    if isinstance(response_json, dict) and "changes" in response_json:
                        try:
                            st.write(response_json["markdown"])
                            st.write(response_json["summary"])
                        except:
                            st.json(response_json)

                        st.json(response_json)
                        logger.info(f"Answer Question Method: {response_json}")
                        with st.spinner("Implementing changes..."):
                            results = st.session_state.rag_app.implement_changes(
                                response_json["changes"]
                            )
                            if results != []:
                                st.write("#### Implementation")
                                st.json(results, expanded=False)
                            if any(result["status"] == "error" for result in results):
                                st.error(
                                    "Some changes could not be implemented. Check the results for details."
                                )
                            else:
                                st.success("All changes implemented successfully!")
                except json.JSONDecodeError:
                    st.error(
                        "The response is not in valid JSON format. No changes to implement."
                    )

        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    else:
        st.warning("Please initialize the project first using the sidebar.")

# Clear chat history button
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    if st.session_state.rag_app:
        st.session_state.rag_app.chat_history = []
    st.success("Chat history cleared!")
