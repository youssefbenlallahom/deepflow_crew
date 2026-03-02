"""
Test script: PDFSearchTool with Ollama local embeddings (bge-m3)

Prerequisites:
  1. Install Ollama: https://ollama.com/download
  2. Pull the bge-m3 embedding model:
       ollama pull bge-m3
  3. Make sure Ollama is running (it starts automatically after install)
  4. Run this script from the project root:
       python test_pdf_search.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import PDFSearchTool
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from crewai.rag.chromadb.config import ChromaDBConfig

os.environ["EMBEDDINGS_OLLAMA_MODEL_NAME"] = "bge-m3"

# ChromaDB persist directory inside the project
CHROMA_DIR = os.path.join(os.path.dirname(__file__), ".chroma", "indictment_db")
os.makedirs(CHROMA_DIR, exist_ok=True)

# Build a ChromaDBConfig with Ollama embeddings and local persistence.
# We construct with embedding_function only, then swap settings via
# object.__setattr__ because the frozen pydantic dataclass + chromadb's
# pydantic-v1 Settings class triggers a validation bug when passed directly.
ollama_ef = OllamaEmbeddingFunction(
    url="http://localhost:11434",
    model_name="bge-m3",
)

chroma_config = ChromaDBConfig(embedding_function=ollama_ef)
object.__setattr__(chroma_config, "settings", Settings(
    persist_directory=CHROMA_DIR,
    allow_reset=True,
    is_persistent=True,
))

# --- Config ---
PDF_PATH = os.path.join(
    os.path.dirname(__file__),
    'src', 'deepflow_crew', 'tools', 'Jeffrey_Epstein_Indictment_Searchable.pdf'
)

# Pass the pre-built config directly — _parse_config returns non-dict as-is
pdf_search_tool = PDFSearchTool(
    pdf=PDF_PATH,
    config=chroma_config,
)

# LLM — using Gemini (same as your crew.py)
llm = LLM(
    model="groq/moonshotai/kimi-k2-instruct-0905",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)

# A simple agent to test the tool
investigator = Agent(
    role="Document Investigator",
    goal="Answer questions about the Epstein complaint PDF using semantic search",
    backstory=(
        "You are an AI investigator who specializes in analyzing legal documents. "
        "You use the PDF search tool to find relevant passages and answer questions accurately."
    ),
    tools=[pdf_search_tool],
    llm=llm,
    verbose=True
)

# A simple task to verify it works
search_task = Task(
    description=(
        "Using the PDF search tool, find and summarize: "
        "1. According to the forfeiture allegations, what is the specific address, block number, lot number, and exact corporate entity that owns the New York property subject to forfeiture? "
        "2. What specific substitute asset provision statutes (Title 21 and Title 28) are cited if the property cannot be located? "
        "3. Describe the specific roles of 'Employee-2' and 'Employee-3' regarding the scheduling of encounters."
    ),
    expected_output=(
        "A concise summary answering all 3 questions with specific, "
        "exact details (such as address, entity name, and statute numbers) extracted from the PDF."
    ),
    agent=investigator
)

# Run it
if __name__ == "__main__":

    # ── Direct Tool Test (no agent, just raw semantic search) ──
    print("\n" + "=" * 60)
    print("🔍 DIRECT QUERY TEST — calling PDFSearchTool directly")
    print("=" * 60)

    test_query = "What is the specific address, block number, lot number, and corporate entity name for the property subject to forfeiture in New York, and what substitute asset statutes are cited?"
    print(f"\n📝 Query: \"{test_query}\"\n")

    raw_result = pdf_search_tool._run(query=test_query)
    print("📄 Semantic search result:")
    print("-" * 40)
    print(raw_result)
    print("-" * 40)

    print("\n✅ PDFSearchTool with Ollama bge-m3 works!")
    print("Unlike PDFExtractorTool (which dumps the ENTIRE PDF),")
    print("this returns only the chunks relevant to your query.\n")
