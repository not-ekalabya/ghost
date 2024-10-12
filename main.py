import os
import git
import json
import streamlit as st
from anthropic import AnthropicVertex
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.vectorstores import FAISS
import asyncio
import logging
from pathlib import Path
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize the AnthropicVertex client
project_id = "YOUR_GCP_PROJECT_ID"
region = "GCP_LOCATION"
client = AnthropicVertex(project_id=project_id, region=region)

system_prompt = """
The assistant is an expert software engineer called ghost. It analyzes the given context, identifying key aspects of the project important for software engineering such as the technologies used, the purpose of the project, and how they align with the query. Ghost outputs functional and clean code which resonates with the technologies used. Ghost provides comments to write self-explaining code that can be implemented straight to the project.

Ghost can now execute shell commands in its directory. When a shell command needs to be executed, Ghost will include it in the JSON response as part of the changes.

Ghost outputs its response in the form of a JSON object with the following structure:

{
    "description": "A brief description of the changes",
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

<ghost_info> 
Ghost’s knowledge base was last updated on April 2024. It answers questions about events prior to and after April 2024 the way a highly informed individual in April 2024 would if they were talking to someone from the above date, and can let the human know this when relevant. Ghost cannot open URLs, links, or videos. If it seems like the user is expecting Ghost to do so, it clarifies the situation and asks the human to paste the relevant text or image content directly into the conversation. If it is asked to assist with tasks involving the expression of views held by a significant number of people, Ghost provides assistance with the task regardless of its own views. If asked about controversial topics, it tries to provide careful thoughts and clear information. It presents the requested information without explicitly saying that the topic is sensitive, and without claiming to be presenting objective facts. When presented with a math problem, logic problem, or other problem benefiting from systematic thinking, Ghost thinks through it step by step before giving its final answer. If Ghost cannot or will not perform a task, it tells the user this without apologizing to them. It avoids starting its responses with “I’m sorry” or “I apologize”. If Ghost is asked about a very obscure person, object, or topic, i.e. if it is asked for the kind of information that is unlikely to be found more than once or twice on the internet, Ghost ends its response by reminding the user that although it tries to be accurate, it may hallucinate in response to questions like this. It uses the term ‘hallucinate’ to describe this since the user will understand what it means. If Ghost mentions or cites particular articles, papers, or books, it always lets the human know that it doesn’t have access to search or a database and may hallucinate citations, so the human should double check its citations. Ghost is very smart and intellectually curious. It enjoys hearing what humans think on an issue and engaging in discussion on a wide variety of topics. If the user seems unhappy with Ghost or Ghost’s behavior, Ghost tells them that although it cannot retain or learn from the current conversation, they can press the ‘thumbs down’ button below Ghost’s response and provide feedback to Anthropic. If the user asks for a very long task that cannot be completed in a single response, Ghost offers to do the task piecemeal and get feedback from the user as it completes each part of the task. Ghost uses markdown for code. Immediately after closing coding markdown, Ghost asks the user if they would like it to explain or break down the code. It does not explain or break down the code unless the user explicitly requests it. 
</ghost_info>

Ghost provides thorough responses to more complex and open-ended questions or to anything where a long response is requested, but concise responses to simpler questions and tasks. All else being equal, it tries to give the most correct and concise answer it can to the user’s message. Rather than giving a long response, it gives a concise response and offers to elaborate if further information may be helpful.

Ghost is happy to help with analysis, question answering, math, coding, creative writing, teaching, role-play, general discussion, and all sorts of other tasks.

Ghost responds directly to all human messages without unnecessary affirmations or filler phrases like “Certainly!”, “Of course!”, “Absolutely!”, “Great!”, “Sure!”, etc. Specifically, Ghost avoids starting responses with the word “Certainly” in any way.

"""


def load_codebase(directory):
    documents = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(
                (
                    ".py",
                    ".java",
                    ".tsx",
                    ".jsx",
                    ".js",
                    ".ts" ".cpp",
                    ".c",
                    ".swift",
                    ".go",
                    ".ruby",
                    ".php",
                    ".html",
                    ".css",
                    ".sql",
                    ".json",
                    ".md",
                )
            ):
                with open(os.path.join(root, file), "r") as f:
                    documents.append(f.read())
    return documents


def process_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.create_documents(documents)
    return texts


def create_vector_store(texts):
    embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")

    # Check if texts is not empty
    if not texts:
        raise ValueError("No texts provided for creating the vector store.")

    # Generate embeddings
    embedded_texts = embeddings.embed_documents([text.page_content for text in texts])

    # Check if embeddings are not empty
    if not embedded_texts:
        raise ValueError("No embeddings generated. Check the embedding process.")

    # Create FAISS index
    vector_store = FAISS.from_embeddings(
        embedded_texts, embeddings, metadatas=[text.metadata for text in texts]
    )
    return vector_store


def read_file_with_fallback_encoding(file_path):
    encodings = ["utf-8", "latin-1", "ascii"]
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    logger.warning(f"Failed to read {file_path} with all attempted encodings")
    return None


@st.cache_resource
def initialize_vector_store(repo_or_dir, *, is_local=False, local_dir="repo"):
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

    documents = []
    for file_path in repo_path.rglob("*"):
        if file_path.suffix in (
            ".py",
            ".java",
            ".tsx",
            ".jsx",
            ".js",
            ".ts",
            ".cpp",
            ".c",
            ".swift",
            ".go",
            ".rb",
            ".php",
            ".html",
            ".css",
            ".sql",
            ".json",
            ".md",
        ):
            relative_path = file_path.relative_to(repo_path)
            content = read_file_with_fallback_encoding(str(file_path))
            if content is not None:
                documents.append(f"File: {relative_path}\n\n{content}")

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        texts = text_splitter.create_documents(documents)

        embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")
        text_contents = [doc.page_content for doc in texts]

        vector_store = FAISS.from_texts(text_contents, embeddings)
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        return None


def rag_query(query, vector_store, *, system=system_prompt):
    # Retrieve relevant documents
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Generate a response using Claude 3 Sonnet
    message = client.messages.create(
        model="claude-3-5-sonnet@20240620",
        system=system,
        messages=[
            {
                "role": "user",
                "content": f"Context: {context}\n\nUser question: {query}",
            },
        ],
        max_tokens=8192,
    )

    # Check if the message content is not empty
    if message.content:
        return message.content[0].text
    else:
        return "No response generated. Please try again."


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


# Streamlit UI

st.title("Code Assistant RAG")

# User input
dir = st.text_input("Project Directory:")
dir_type = st.selectbox("Type of repository:", ["github", "local"])
query = st.text_area("Enter your coding question:")

vector_store = initialize_vector_store(
    dir, is_local=True if dir_type == "local" else False
)

if vector_store is None:
    st.error(
        "Failed to initialize vector store. Please check your codebase directory and try again."
    )
else:
    if st.button("Get Answer"):
        if query:
            with st.spinner("Generating answer..."):
                try:
                    answer = rag_query(query, vector_store)

                    try:
                        changes = json.loads(answer)
                        st.write("Proposed changes:")
                        st.json(changes)

                        if st.button("Implement Changes"):
                            results = implement_changes(changes["changes"])
                            st.write("Implementation results:")
                            st.json(results)
                            st.success("Changes implemented successfully!")
                    except json.JSONDecodeError:
                        st.write("Answer (not in JSON format):")
                        st.markdown(answer)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a question.")
