# DeepFlow Crew Workshop and Candidate Exercise

This repository is a CrewAI workshop project with a two-agent pipeline.
Current baseline is intentionally simple: PDF-only evidence collection.

## 1) Dependencies and Requirements

### Python and Tooling
- Python: `>=3.10,<3.13`
- Package manager/environment: `uv`
- Crew runner: `crewai`

### Project Dependencies
From `pyproject.toml`:
- `crewai[tools]>=0.121.0,<1.0.0`
- `chromadb>=1.1.1`
- `ollama>=0.6.1`
- `pypdf2>=3.0.0`

### Local Services and Keys
- Ollama running locally on `http://localhost:11434`
- Embedding model available in Ollama: `bge-m3`
- `.env` must include: `OPENROUTER_API_KEY`

### Install and Run
```bash
crewai install
crewai run
```

---

## 2) What Exists in the Workshop (Current Baseline)

### Agents
- `investigative_researcher`: collects evidence from the PDF only.
- `legal_analyst`: writes the final report from the researcher output.

### Tasks
- `research_task`: PDF-grounded evidence brief with required citations and tool execution log.
- `reporting_task`: final 600–900 word legal-style report, based on research brief.

### Tooling Strategy
- Researcher has only `PDFSearchTool`.
- Analyst is intentionally tool-less (synthesis role).

---

## 3) Workflow Step by Step

1. `crewai run` starts the crew defined in `crew.py`.
2. `research_task` is assigned to `investigative_researcher`.
3. Researcher queries the indictment PDF using `PDFSearchTool`.
4. Researcher returns a structured evidence brief:
	- findings,
	- PDF page/section citations,
	- named entities,
	- contradictions/open questions,
	- `Tool Execution Log`.
5. `reporting_task` is assigned to `legal_analyst`.
6. Analyst consumes the evidence brief and produces the final report in `epstein_files_report.md`.

This baseline demonstrates agent specialization and handoff before introducing multi-tool web corroboration.

---

## 4) Candidate Exercise (GitHub Assignment)

### Goal
Extend the workshop so the researcher combines PDF evidence with web corroboration.

### Candidate Tasks
1. In `src/deepflow_crew/crew.py`, add `SerperDevTool` and `ScrapeWebsiteTool` to `investigative_researcher` tools.
2. In `src/deepflow_crew/config/tasks.yaml`, update `research_task` prompt:
	- require PDF evidence and web corroboration,
	- require at least one web-backed finding,
	- keep `Tool Execution Log` mandatory.
3. Keep `legal_analyst` tool-less.
4. Keep the same report structure and word range for `reporting_task`.

### Acceptance Criteria
- Crew logs show real tool calls (PDF + at least one web tool).
- Research output includes `Tool Execution Log` with concrete queries/URLs.
- Final report remains source-backed and structured.
- No hallucinated links presented as verified evidence.

### Submission Checklist
- Short note describing code changes.
- One run snippet showing tool calls.
- Final generated `epstein_files_report.md`.

---

## 5) Suggested Evaluation Rubric

- Correctness of tool integration (30%)
- Quality of evidence and citations (30%)
- Prompt/task design clarity (20%)
- Clean handoff to reporting agent (20%)

---

## 6) References

- CrewAI Docs: https://docs.crewai.com
- CrewAI GitHub: https://github.com/crewAIInc/crewAI
