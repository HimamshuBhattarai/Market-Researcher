# Market Research Brief

A Streamlit app that turns a company or industry question (for example "Analyze the food delivery market in Nepal.") into a short research brief. It searches the web, extracts findings with quoted evidence, drafts a brief and runs one automated review. The brief contains a summary, up to two competitors, key observations, sources, and a list of what could not be found.

## How it works

Four roles run in sequence and pass structured (Pydantic) data to each other:

1. **Orchestrator** receives the query and runs the workflow (a LangGraph graph).
2. **Researcher** plans questions, searches the web (Tavily) and extracts findings that quote their source.
3. **Analyst** writes the brief using only those findings.
4. **Critic** checks the brief against the cited evidence and can request **one** revision. The revised brief is final.

## Setup

You need Python 3.13 or newer, [uv](https://docs.astral.sh/uv/getting-started/installation/), and two free API keys: one from [Groq](https://console.groq.com/keys) and one from [Tavily](https://app.tavily.com/home).

```bash
git clone https://github.com/HimamshuBhattarai/Market-Researcher
cd Market-Researcher
uv venv          
.venv/Scripts/activate # Activates the venv
uv sync          # installs the dependencies into it
```

Create your `.env` file from the template and add your two keys:

```bash
cp .env.example .env
```

```
GROQ_API_KEY=your-groq-key
SEARCH_API_KEY=your-tavily-key
```

Optional settings: `GROQ_MODEL` (default `openai/gpt-oss-20b`), and `LLM_PROVIDER=ollama` with `OLLAMA_MODEL` to use a local Ollama model instead of Groq (weaker results). Code supports for both the LLM provider(we can add a fallback API, but I finished all my credits for Free-tier API)

## Run

```bash
cd backend/app
uv run streamlit run app.py
```

Open http://localhost:8501, enter a company or industry, and click **Run research**. A run takes a few minutes and shows the current stage. Stop with Ctrl+C.

For running backend

```bash
cd backend/app
uv run uvicorn main:app --reload
```

Documentation is at http://127.0.0.1:8000/docs, and the endpoint is `POST /api/research` with a body such as `{"query": "Analyze the food delivery market in Nepal."}`.


## Tests

```bash
cd backend/app
uv run pytest
```

The tests run offline and use no API credits.

## How the brief is checked

- Figures in the brief must appear in the findings they cite, and each finding must quote its source.
- Findings are labelled as about the target place or as broader context, and weak sources are attributed ("According to ...").
- Only rule-based checks can trigger the one revision. The model reviewer's remarks are advisory.
- This does not guarantee accuracy. Verify important claims against the linked sources.

## Limitations

- Public data for some markets, including Nepal, is scarce. Gaps are reported under "Missing information" instead of being invented.
- The competitor list is empty when no cited evidence names a company.
- Results vary between runs. Free API tiers have daily limits, and one run makes roughly 25 to 35 model calls plus several searches.

## Project structure

```
backend/app/app.py                    Streamlit page
backend/app/main.py                   FastAPI app
backend/app/market_research/          agents, research, schemas, services, config
backend/app/tests/                    offline tests
.env.example                          configuration template
```

## Note

I built this within a short deadline and tested it mainly on the food delivery market in Nepal and India, so it may have flaws I did not catch. If something does not work as described here, please let me know.