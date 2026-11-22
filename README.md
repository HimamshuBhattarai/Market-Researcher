Market Research Brief

Market Research Brief is a Streamlit application that turns a company or market
question into a short evidence-backed research brief. It searches public
sources, extracts findings, drafts a brief and applies one automated review.

The FastAPI API exposes the same workflow for clients that prefer a separate
backend process. The local page runs the workflow in-process unless an API URL
is explicitly configured.

## How it works

```text
Orchestrator
    -> Researcher
    -> Analyst
    -> Critic
       | approved or no rule-based issue
       -> Completed
       | rule-based issue
       -> One revision
       -> Completed
```

The orchestrator receives the query and starts the LangGraph workflow. Pydantic
models pass the research findings, draft, critic feedback and final state
between the roles. A revised draft is not sent back through the critic.

## Requirements

- Python 3.13 or newer
- uv, or pip as an alternative for local installation
- A Groq API key and Tavily API key for the recommended hosted setup
- Ollama as an optional local language-model provider instead of Groq

Groq is the main and default language-model provider. Tavily is the only search
provider implemented by the current code. Ollama is also supported, so users
who prefer a local model can change `LLM_PROVIDER=ollama` in `.env`.

## Setup

From the repository root:

```powershell
git clone <repository-url>
cd Market-Researcher
uv sync
copy .env.example .env
```

Fill in the values in `.env`. `.env` is local configuration and must not be
committed.

With pip, the alternative is:

```powershell
python -m pip install -e .
python -m pip install pytest
```

## Configuration

| Variable | Allowed values | Default | Meaning |
|---|---|---|---|
| `LLM_PROVIDER` | `groq`, `ollama` | `groq` | Selects the language-model provider |
| `GROQ_API_KEY` | API key text | none | Primary Groq API key |
| `groq` | API key text | none | Legacy fallback Groq key name |
| `GROQ_MODEL` | Groq model name | `openai/gpt-oss-20b` | Groq model used for structured calls |
| `LLM_JSON_MODE` | `auto`, `on`, `off` | `auto` | Controls JSON response formatting |
| `TAVILY_API_KEY` | API key text | none | Primary Tavily search key |
| `SEARCH_API_KEY` | API key text | none | Legacy fallback search key alias |
| `OLLAMA_MODEL` | Ollama model name | none | Required when `LLM_PROVIDER=ollama` |
| `OLLAMA_BASE_URL` | URL | `http://localhost:11434/v1` | Local Ollama OpenAI-compatible endpoint |
| `MARKET_RESEARCH_API_URL` | URL or unset | unset | If set, Streamlit calls this FastAPI URL instead of running the workflow in-process |

## Language-model providers

Groq is the main and default provider. Create a Groq account and set
`GROQ_API_KEY`, `GROQ_MODEL` and `TAVILY_API_KEY` in `.env`.

To use Ollama locally, install Ollama and set:

```text
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434/v1
```

Then install the configured model:

```powershell
ollama pull llama3.1:8b
```

Small local models can give weaker or less consistent structured output. Set
the Ollama server context window larger for longer prompts if needed, for
example:

```powershell
$env:OLLAMA_CONTEXT_LENGTH="8192"
ollama serve
```

Free hosted tiers can have daily limits or quota restrictions. A live run may
make roughly 25 to 35 model calls and several search requests. The project uses
the `openai` Python package as a generic OpenAI-compatible client; an OpenAI
account is not required.

## Run the Streamlit app

From `backend/app`:

```powershell
cd backend/app
uv run streamlit run app.py
```

Open `http://localhost:8501`, enter a company or industry, and click
`Run research`. Stop Streamlit with Ctrl+C.

By default, the page imports and runs the workflow directly. It shows live stage
progress while the workflow runs. If `MARKET_RESEARCH_API_URL` is set, the page
instead sends the request to that API with a ten-minute timeout.

## Run the optional API

From `backend/app`:

```powershell
cd backend/app
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

Open the API documentation at `http://127.0.0.1:8000/docs`.

The research endpoint is `POST /api/research`. It accepts:

```json
{
  "query": "Analyze the food delivery market in Nepal.",
  "research_path": null
}
```

Set `MARKET_RESEARCH_API_URL=http://127.0.0.1:8000` to make Streamlit call this
API instead of running the workflow in-process.

## Example query

```text
Analyze the food delivery market in Nepal.
```

The output contains an executive summary, up to two cited competitors,
key observations, evidence, sources and gaps or limitations.

## Sample output

Sample files are stored in [backend/app/samples](backend/app/samples):

- `sample_output.md` records that the brief was produced from resumed saved
  research.
- `sample_output.json` records `"research_mode": "resumed"`.
- `sample_output_nepal.md` is a Nepal brief, but its source mode is not recorded
  in the file and the original source mode should be confirmed by the author.
- JSON source access dates end at 2026-11-21; the Nepal Markdown file shows
  source access dates of 2026-11-22.

These files are examples rather than generated guarantees. The sample folder
is not ignored and can be shipped in the repository.

## Tests

From the repository root:

```powershell
uv run python backend/app/tests/main.py
```

The tests use mocks and do not call the LLM or search providers.

## How briefs are checked

- Figures in the brief must appear in their cited findings.
- Findings are labelled as target geography or broader context.
- Weak sources and opinions are attributed cautiously.
- Cited attributions are checked against source publishers.
- The critic can trigger at most one revision.
- Only rule-based critic checks trigger a revision.
- The language-model critic findings are advisory and are not used to force a
  revision.
- The workflow does not guarantee that every statement or source is accurate.
  Important claims should be verified against the original sources.

## Known limitations

- Public evidence for some markets, including Nepal, can be scarce.
- Missing market metrics are reported instead of being invented.
- The competitor list can be empty when cited evidence does not name a company.
- Search and language-model results can vary between runs and models.
- Geography detection relies on explicit geography text and a place-name list
  that includes Nepal and India.
- The language-model reviewer can miss unsupported statements.
- Search and language-model free-tier quotas can stop a run.
- A live run spends Tavily search credits and hosted model usage.

## Project structure

```text
backend/app/app.py                         Streamlit page
backend/app/main.py                        FastAPI application
backend/app/router/research.py             Research API route
backend/app/market_research/agents/        Orchestrator, researcher, analyst, critic
backend/app/market_research/research/      Planner and Tavily search
backend/app/market_research/schemas/       Pydantic workflow, brief and source models
backend/app/market_research/services/llm.py OpenAI-compatible LLM client
backend/app/samples/                       Shipped sample briefs
backend/app/tests/                         Offline pytest suite
backend/app/.streamlit/config.toml         Streamlit theme
.env.example                               Environment template
pyproject.toml                             Python project and dependencies
uv.lock                                    Locked dependencies
```