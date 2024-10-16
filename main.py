import os
import json
import streamlit as st
from anthropic import AnthropicVertex, APIStatusError
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
import base64
import difflib

import pytest
import tempfile
import ast
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import traceback

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
            system="""You are an expert software engineer called ghost. You analyze the given context and provide code modifications.
            Your response must be in VALID JSON format with the following structure:
            {
                "description": "A brief description of the changes",
                "markdown": "explanatory text",
                "changes": [{
                    "action": "create|modify|delete|execute",
                    "path": "path/to/file",
                    "content": "file_content_or_diff",
                    "command": "shell_command"  // for execute action only
                }],
                "summary": "summary of changes"
            }
            For 'modify' actions, encode the diff content to handle newlines and special characters. Use '\\n' for newlines.
            IMPORTANT: All strings in the JSON must be properly escaped.""",
            max_tokens=8192,
            messages=messages,
        )

        # Try to parse the response as JSON
        try:
            json.loads(response.content[0].text)
            return response.content[0].text
        except json.JSONDecodeError:
            # If parsing fails, try to extract JSON portion and fix common issues
            text = response.content[0].text
            try:
                # Find JSON-like content between curly braces
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = text[start:end]
                    # Parse to validate and re-encode properly
                    parsed = json.loads(json_str)
                    # Fix diff content in changes
                    if "changes" in parsed:
                        for change in parsed["changes"]:
                            if "content" in change:
                                # Properly escape newlines and special characters
                                change["content"] = (
                                    change["content"]
                                    .replace("\n", "\\n")
                                    .replace('"', '\\"')
                                )
                    return json.dumps(parsed, ensure_ascii=False)
            except (json.JSONDecodeError, AttributeError) as e:
                logger.error(f"Failed to fix JSON response: {e}")
                # Return a valid JSON error response
                return json.dumps(
                    {
                        "error": "Invalid response format",
                        "description": "Failed to generate valid JSON response",
                        "changes": [],
                        "summary": "No changes implemented due to response format error",
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


@dataclass
class TestResult:
    success: bool
    error_message: str = None
    traceback: str = None
    output: str = None


class TestRunner:
    def __init__(self, codebase_directory: str):
        self.codebase_directory = codebase_directory
        self.temp_test_dir = tempfile.mkdtemp()

    def generate_test_file(self, test_specs: List[dict]) -> str:
        """Generate a pytest file from test specifications"""
        test_content = ["import pytest", "import sys", "import os", ""]

        # Add path to system path to import project modules
        test_content.append(f"sys.path.append('{self.codebase_directory}')")

        for spec in test_specs:
            if spec["type"] == "unit":
                test_content.extend(self._generate_unit_test(spec))
            elif spec["type"] == "integration":
                test_content.extend(self._generate_integration_test(spec))

        test_path = os.path.join(self.temp_test_dir, "test_generated.py")
        with open(test_path, "w") as f:
            f.write("\n".join(test_content))

        return test_path

    def _generate_unit_test(self, spec: dict) -> List[str]:
        """Generate a unit test from specification"""
        test_lines = []
        test_lines.append(f"\n\ndef test_{spec['name']}():")

        # Add imports
        if "imports" in spec:
            for imp in spec["imports"]:
                test_lines.append(f"    {imp}")

        # Add setup
        if "setup" in spec:
            test_lines.extend(f"    {line}" for line in spec["setup"])

        # Add assertions
        for assertion in spec["assertions"]:
            test_lines.append(f"    {assertion}")

        return test_lines

    def run_tests(self, test_path: str) -> TestResult:
        """Run the generated tests and return results"""
        try:
            # Run pytest programmatically
            import pytest

            result = pytest.main(["-v", test_path])

            if result == 0:  # All tests passed
                return TestResult(success=True, output="All tests passed successfully")
            else:
                return TestResult(
                    success=False,
                    error_message="Some tests failed",
                    output=f"Pytest exit code: {result}",
                )
        except Exception as e:
            return TestResult(
                success=False, error_message=str(e), traceback=traceback.format_exc()
            )

    def validate_code_statically(self, code: str) -> TestResult:
        """Perform static code analysis"""
        try:
            ast.parse(code)  # Check if code is syntactically valid
            return TestResult(success=True)
        except SyntaxError as e:
            return TestResult(
                success=False,
                error_message=f"Syntax error: {str(e)}",
                traceback=traceback.format_exc(),
            )


class RAGApplication:

    def __init__(self, codebase_directory: str, project_id: str, region: str):
        self.embedder = TextEmbedder()
        self.code_snippets = process_codebase(codebase_directory)
        self.codebase_embeddings = CodebaseEmbeddings(self.code_snippets, self.embedder)
        self.claude_client = ClaudeClient(project_id, region)
        self.codebase_directory = codebase_directory
        self.chat_history: List[ChatMessage] = []
        logger.info(
            f"Initialized RAGApplication with codebase directory: {self.codebase_directory}"
        )

    def answer_question(
        self, question: str, images: List[Tuple[str, bytes]] = None, attempts=3
    ) -> str:
        max_attempts = attempts
        attempt = 0
        last_response_json = None

        while attempt < max_attempts:
            try:
                # Get initial response
                response = self.claude_client.generate_response(
                    [{"role": "user", "content": [{"type": "text", "text": question}]}]
                )
                response_json = json.loads(response)
                last_response_json = response_json  # Store the last response

                # Execute the program
                execution_result = self._execute_program(response_json)

                if execution_result.success:
                    return json.dumps(response_json)

                # If execution failed, get fixes
                fix_prompt = self._generate_fix_prompt(response_json, execution_result)
                response = self.claude_client.generate_response(
                    [{"role": "user", "content": fix_prompt}]
                )

                attempt += 1

            except Exception as e:
                logger.error(f"Error in execution cycle: {str(e)}")
                execution_result = TestResult(
                    success=False,
                    error_message=str(e),
                    traceback=traceback.format_exc(),
                )
                attempt += 1

        # If we've exhausted attempts, return the last response with error info
        final_response = last_response_json or {
            "description": "Failed to process request",
            "changes": [],
            "summary": "Error occurred during processing",
        }

        final_response["execution_status"] = {
            "success": False,
            "error": "Maximum retry attempts reached",
            "last_execution_results": (
                execution_result.__dict__
                if execution_result
                else {
                    "success": False,
                    "error_message": "No execution results available",
                    "traceback": None,
                    "output": None,
                }
            ),
        }

        return json.dumps(final_response)

    def _execute_program(self, response_json: dict) -> TestResult:
        """Execute the program and return the result"""
        execute_command = None
        for change in response_json["changes"]:
            if change["action"] == "execute":
                execute_command = change["command"]
                break

        if not execute_command:
            return TestResult(
                success=False,
                error_message="No execute command found in the response",
            )

        try:
            result = subprocess.run(
                execute_command,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                cwd=self.codebase_directory,
            )
            return TestResult(success=True, output=result.stdout)
        except subprocess.CalledProcessError as e:
            return TestResult(
                success=False,
                error_message=f"Command failed with exit code {e.returncode}",
                traceback=e.stderr,
                output=e.stdout,
            )

    def _generate_fix_prompt(
        self, response_json: dict, execution_result: TestResult
    ) -> str:
        """Generate prompt for fixing failed execution"""
        return f"""The following changes failed execution:

Changes:
{json.dumps(response_json['changes'], indent=2)}

Execution Results:
Error: {execution_result.error_message}
Traceback: {execution_result.traceback}
Output: {execution_result.output}

Please provide fixed changes that will execute successfully. Respond in the same JSON format as before."""

    def parse_unified_diff(self, diff_text: str) -> List[Tuple[str, str, int]]:
        """Parse a unified diff format string into a list of changes."""
        # Handle escaped newlines in the diff text
        diff_text = diff_text.replace("\\n", "\n")
        changes = []
        current_line = 0

        for line in diff_text.splitlines():
            if line.startswith("@@"):
                try:
                    header = line.split("@@")[1].strip()
                    _, new = header.split(" ")
                    current_line = int(new.split(",")[0].lstrip("+")) - 1
                    continue
                except (IndexError, ValueError):
                    logger.error(f"Failed to parse diff header: {line}")
                    continue

            if not line.startswith(("+++", "---")):
                if line.startswith("+"):
                    changes.append(("add", line[1:], current_line))
                    current_line += 1
                elif line.startswith("-"):
                    changes.append(("remove", line[1:], current_line))
                else:
                    current_line += 1

        return changes

    def apply_diff(self, file_path: str, diff_content: str) -> str:
        """Apply a diff to a file and return the resulting content"""
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            lines = []

        # Remove trailing newlines while preserving empty lines
        lines = [line.rstrip("\n") + "\n" for line in lines]

        # Parse and apply the changes
        changes = self.parse_unified_diff(diff_content)
        offset = 0  # Track line number changes as we modify the file

        for change_type, content, line_num in changes:
            adjusted_line = line_num + offset
            if change_type == "add":
                if adjusted_line >= len(lines):
                    lines.append(content + "\n")
                else:
                    lines.insert(adjusted_line, content + "\n")
                offset += 1
            elif change_type == "remove":
                if 0 <= adjusted_line < len(lines):
                    lines.pop(adjusted_line)
                    offset -= 1

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
        results = []
        for change in changes:
            action = change["action"]
            try:
                if action == "create":
                    path = os.path.join(self.codebase_directory, change["path"])
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    # Unescape content before writing
                    content = change["content"].replace("\\n", "\n").replace('\\"', '"')
                    with open(path, "w") as f:
                        f.write(content)
                    self.codebase_embeddings.update_embeddings(path, content)
                    results.append(
                        {
                            "action": action,
                            "path": path,
                            "status": "success",
                            "diff": self.generate_diff(path, content),
                        }
                    )

                elif action == "modify":
                    path = os.path.join(self.codebase_directory, change["path"])
                    if not os.path.exists(path):
                        results.append(
                            {
                                "action": action,
                                "path": path,
                                "status": "error",
                                "message": "File not found",
                            }
                        )
                        continue

                    # Unescape the diff content before applying
                    diff_content = (
                        change["content"].replace("\\n", "\n").replace('\\"', '"')
                    )
                    new_content = self.apply_diff(path, diff_content)

                    with open(path, "w") as f:
                        f.write(new_content)

                    self.codebase_embeddings.update_embeddings(path, new_content)
                    results.append(
                        {
                            "action": action,
                            "path": path,
                            "status": "success",
                            "diff": change["content"],  # Keep escaped version for JSON
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
                                "message": "File not found",
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
                logger.error(f"Error implementing change {action}: {str(e)}")
                results.append(
                    {
                        "action": action,
                        "path": change.get("path", ""),
                        "status": "error",
                        "message": str(e),
                    }
                )

        return results

    def reload_database(self):
        self.codebase_embeddings.reload_database(self.codebase_directory)


system_prompt = """
The assistant is an expert software engineer called ghost. It analyzes the given context, identifying key aspects of the project important for software engineering such as the technologies used, the purpose of the project, and how they align with the query. Ghost outputs functional and clean code which resonates with the technologies used. Ghost provides comments to write self-explaining code that can be implemented straight to the project.

Ghost can execute shell commands in its directory when necessary. When a shell command needs to be executed, Ghost will include it in the JSON response as part of the changes.

Ghost can also process and analyze images provided in the conversation. When images are present, Ghost will consider them in the context of the query and provide relevant insights or code modifications based on the image content.

Ghost outputs its response in the form of a JSON object with the following structure:

{
    "description": "A brief description of the response or changes",
    "markdown": "Detailed explanation or response content",
    "changes": [
        {
            "action": "create|modify|delete|execute",
            "path": "path/to/file",
            "content": "New or modified content of the file in diff format",
            "command": "Shell command to execute"
        }
    ],
    "summary": "A summary of the response or changes implemented"
}

The 'changes' array should only be included when code modifications or executions are necessary. For non-code-related queries, the 'changes' array can be empty or omitted.

For code-related queries:
- Use the 'create', 'modify', or 'delete' actions for file operations.
- Use the 'execute' action when a command needs to be run, typically to test or run the program.
- For 'modify' actions, use a diff-like format in the 'content' field.

For non-code-related queries:
- Provide an appropriate response in the 'markdown' field.
- Do not include unnecessary 'execute' actions or empty 'changes' arrays.

GHOST OUTPUTS ONLY IN VALID JSON FORMAT. THE OUTPUT CAN BE READILY PARSED AS A JSON OBJECT.
Any other information except the JSON is not shown.
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

    project_id = "ghost-widget-7000"
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

                response = st.session_state.rag_app.answer_question(
                    prompt, images, max_attempts
                )

                try:
                    response_json = json.loads(response)
                    if isinstance(response_json, dict) and "changes" in response_json:
                        st.write("Proposed changes:")
                        st.json(response_json)
                        with st.spinner("Implementing changes..."):
                            results = st.session_state.rag_app.implement_changes(
                                response_json["changes"]
                            )
                            st.write("Implementation results:")
                            st.json(results)
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
