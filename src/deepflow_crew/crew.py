from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import PDFSearchTool
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from crewai.rag.chromadb.config import ChromaDBConfig
import os
from pathlib import Path
from crewai import LLM
from dotenv import load_dotenv

load_dotenv()

"""llm = LLM(
    model=os.getenv("MODEL"),
    api_key=os.getenv("AZURE_API_KEY"),
    base_url=os.getenv("AZURE_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION", "2024-06-01"),
    temperature=0.1
)"""
"""llm = LLM(
    model="gemini/gemini-2.5-pro",
    api_key=os.getenv("GEMINI_API_KEY"),  # Or set GOOGLE_API_KEY/GEMINI_API_KEY
    temperature=0.1
)"""

llm = LLM(
    model="openrouter/qwen/qwen3-next-80b-a3b-instruct",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1/chat/completions",
    reasoning_effort="high",
    temperature=0.1
)
# ── PDFSearchTool with Ollama bge-m3 embeddings + local ChromaDB ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = str(PROJECT_ROOT / ".chroma" / "indictment_db")
os.makedirs(CHROMA_DIR, exist_ok=True)

PDF_PATH = str(PROJECT_ROOT / "src" / "deepflow_crew" / "tools" / "Jeffrey_Epstein_Indictment_Searchable.pdf")

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

pdf_search_tool = PDFSearchTool(
    pdf=PDF_PATH,
    config=chroma_config,
)

@CrewBase
class DeepflowCrew():
    """DeepflowCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]


    @agent
    def investigative_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['investigative_researcher'], # type: ignore[index]
            tools=[pdf_search_tool],
            verbose=True,
            reasoning=True,
            llm=llm
        )

    @agent
    def legal_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['legal_analyst'],
            llm=llm,
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'], # type: ignore[index]
            output_file='epstein_files_report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the DeepflowCrew crew"""
        
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
