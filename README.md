<div align="center">

<img src="Asset 60.png" alt="DeepFlow — AI For All" width="280"/>

<br/>

# DeepFlow Crew Workshop

**Intro to AI Agents powered by CrewAI**

[![CrewAI](https://img.shields.io/badge/Framework-CrewAI-blue?style=for-the-badge)](https://crewai.com)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-yellow?style=for-the-badge&logo=python)](https://python.org)
[![Ollama](https://img.shields.io/badge/Embeddings-Ollama_bge--m3-green?style=for-the-badge)](https://ollama.com)

*A hands-on workshop by the **DeepFlow** club — AI For All.*

</div>

---

## 1 · Dependencies and Requirements

### Python and Tooling
| Requirement | Version / Note |
|---|---|
| Python | `>=3.10, <3.13` |
| Package manager | [`uv`](https://docs.astral.sh/uv/) |
| Crew runner | `crewai` CLI |

### Project Dependencies (from `pyproject.toml`)
| Package | Purpose |
|---|---|
| `crewai[tools]>=0.121.0,<1.0.0` | Agent framework + built-in tools |
| `chromadb>=1.1.1` | Local vector store for PDF RAG |
| `ollama>=0.6.1` | Local embedding model client |
| `pypdf2>=3.0.0` | PDF text extraction |

### Local Services and Keys
- **Ollama** running on `http://localhost:11434` with model `bge-m3` pulled.
- `.env` must include:
  ```
  OPENROUTER_API_KEY=your_key_here
  ```

### Install and Run
```bash
crewai install
crewai run
```

---

## 2 · What Exists in the Workshop (Current Baseline)

### Agents

| Agent | Role | Tools |
|---|---|---|
| `investigative_researcher` | Extracts evidence from the PDF | `PDFSearchTool` |
| `legal_analyst` | Writes the final structured report | *None (synthesis only)* |

### Tasks

| Task | Agent | Purpose |
|---|---|---|
| `research_task` | `investigative_researcher` | PDF-grounded evidence brief with citations and tool execution log |
| `reporting_task` | `legal_analyst` | 600–900 word legal-style Markdown report based on the evidence brief |

### Why the analyst has no tools
An agent in CrewAI is **not** just an LLM call — it has a role, goal, backstory, task context, and collaboration behavior. Tools are optional capabilities. A tool-less agent is a valid **synthesis agent** that demonstrates the difference between a raw LLM and an orchestrated agent.

---

## 3 · Workflow Step by Step

```
crewai run
    │
    ▼
┌──────────────────────────────────────┐
│  1. research_task                    │
│     Agent: investigative_researcher  │
│     Tool:  PDFSearchTool             │
│                                      │
│     → queries indictment PDF         │
│     → returns evidence brief:        │
│       • 8-12 findings + citations    │
│       • named entities               │
│       • contradictions               │
│       • Tool Execution Log           │
└──────────────┬───────────────────────┘
               │ evidence brief passed
               ▼
┌──────────────────────────────────────┐
│  2. reporting_task                   │
│     Agent: legal_analyst             │
│     Tools: none                      │
│                                      │
│     → synthesizes evidence brief     │
│     → writes final report:           │
│       Introduction, Key Figures,     │
│       Timeline, Legal Proceedings,   │
│       Conclusion                     │
│     → saves to epstein_files_report.md│
└──────────────────────────────────────┘
```

---

## 4 · Candidate Exercise

### Goal
Extend the researcher so it combines **PDF evidence** with **live web corroboration** using two additional CrewAI tools.

### What You Need to Do

**Step 1 — Add tools in code**
In `src/deepflow_crew/crew.py`, add `SerperDevTool` and `ScrapeWebsiteTool` to the `investigative_researcher` tools list.

**Step 2 — Update the task prompt**
In `src/deepflow_crew/config/tasks.yaml`, update `research_task` so that:
- web corroboration is required in addition to PDF evidence,
- at least one finding must be web-source-backed,
- `Tool Execution Log` remains mandatory.

**Step 3 — Do not touch the analyst**
Keep `legal_analyst` tool-less. Keep the same report structure and word range for `reporting_task`.

### Acceptance Criteria
- [ ] Crew logs show real tool calls (PDF + at least one web tool).
- [ ] Research output includes `Tool Execution Log` with concrete queries/URLs.
- [ ] Final report remains source-backed and structured.
- [ ] No hallucinated links presented as verified evidence.

### Submission Checklist
- [ ] Short note describing your code changes.
- [ ] One run snippet showing tool calls in the terminal output.
- [ ] Final generated `epstein_files_report.md`.

---

## 5 · Helpful Links for Tool Integration

| Resource | Link |
|---|---|
| CrewAI Tools Overview | https://docs.crewai.com/concepts/tools |
| SerperDevTool (web search) | https://docs.crewai.com/tools/serperapitool |
| ScrapeWebsiteTool | https://docs.crewai.com/tools/scrapewebsitetool |
| PDFSearchTool (already used) | https://docs.crewai.com/tools/pdfsearchtool |
| Creating Custom Tools | https://docs.crewai.com/concepts/tools#creating-your-own-tools |
| CrewAI GitHub | https://github.com/crewAIInc/crewAI |

---

<div align="center">

<img src="Asset 60.png" alt="DeepFlow" width="100"/>

<br/>

**DeepFlow — AI For All**

*Built with curiosity. Powered by agents.*

</div>
