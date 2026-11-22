import logging
from pathlib import Path
from typing import Callable, Literal

from market_research.agents.analyst import run_analyst
from market_research.agents.critic import run_critic
from market_research.agents.researcher import run_researcher
from market_research.agents.researcher import apply_loaded_research_rules
from market_research.config import ConfigError
from market_research.schemas import ResearcherOutput, WorkflowState


logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str], None]
PipelineStatus = Literal[
    "researching",
    "analysing",
    "reviewing",
    "revising",
    "completed",
    "failed",
]


def _progress(
    callback: ProgressCallback | None,
    message: str,
    state: WorkflowState,
    status: PipelineStatus,
) -> None:
    state.status = status
    if callback:
        callback(message)


def _failure_message(error: Exception) -> str:
    if isinstance(error, ConfigError):
        return "Configuration is incomplete. Fill in .env with the required API settings."
    detail = str(error).lower()
    if "404" in detail or "model" in detail and "does not exist" in detail:
        return "The configured Groq model is unavailable or unsupported. Set GROQ_MODEL in .env to a model available in your Groq account."
    if "413" in detail or "too large" in detail or "tokens per minute" in detail:
        return "The Groq request exceeded the account's free-plan token limit. Please try again or reduce GROQ message size."
    if "empty response" in detail:
        return "Groq returned an empty response. Please retry; the model may be temporarily unavailable or overloaded."
    if "503" in detail or "unavailable" in detail or "high demand" in detail:
        return "Groq is temporarily unavailable due to high demand. Please try again shortly."
    if "429" in detail or "resource_exhausted" in detail:
        return "Groq request quota was temporarily exceeded. Please try again later."
    return "The market research run could not be completed. Please check the configuration and try again."


def _mark_resumed_brief(state: WorkflowState, resumed: bool) -> None:
    if resumed and state.final_brief:
        state.final_brief.missing_information = [
            "This brief was produced from a saved research dataset, not from a new live search.",
            *[
                item
                for item in state.final_brief.missing_information
                if item != "This brief was produced from a saved research dataset, not from a new live search."
            ],
        ]


def run_pipeline(
    query: str,
    on_progress: ProgressCallback | None = None,
    research_path: str | Path | None = None,
) -> WorkflowState:
    """Run research or resume it, then analysis, review, and revision."""
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    state = WorkflowState(query=query)
    resumed = research_path is not None

    if research_path:
        try:
            path = Path(research_path)
            state.research_output = ResearcherOutput.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            state.research_output = apply_loaded_research_rules(
                state.research_output
            )
            if on_progress:
                on_progress(f"MODE: RESUMED from saved research ({path})")
        except Exception as error:
            logger.error("Could not load saved research from %s: %s", research_path, error)
            state.status = "failed"
            state.error = "The saved research file could not be loaded or was invalid."
            return state
    else:
        _progress(on_progress, "Stage 1 of 4: Researcher...", state, "researching")
        try:
            state.research_output = run_researcher(query, on_progress)
        except Exception as error:
            logger.error("Researcher failed for query %r: %s", query, error)
            state.status = "failed"
            state.error = _failure_message(error)
            return state

        try:
            output_dir = Path.cwd() / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "last_research.json").write_text(
                state.research_output.model_dump_json(indent=2),
                encoding="utf-8",
            )
        except Exception as error:
            logger.error("Could not save last research output: %s", error)

    _progress(on_progress, "Stage 2 of 4: Analyst...", state, "analysing")
    try:
        state.draft = run_analyst(query, state.research_output)
    except Exception as error:
        logger.error("First analyst call failed for query %r: %s", query, error)
        state.status = "failed"
        state.error = _failure_message(error)
        return state

    _progress(on_progress, "Stage 3 of 4: Critic...", state, "reviewing")
    try:
        state.critic_feedback = run_critic(
            query,
            state.research_output,
            state.draft,
        )
    except Exception as error:
        logger.error("Critic failed for query %r: %s", query, error)
        state.draft.missing_information.append(
            "The automated review could not be completed."
        )
        state.final_brief = state.draft
        _mark_resumed_brief(state, resumed)
        _progress(on_progress, "Stage 4 of 4: Completed...", state, "completed")
        return state

    if state.critic_feedback.approved:
        state.final_brief = state.draft
        _mark_resumed_brief(state, resumed)
        _progress(on_progress, "Stage 4 of 4: Completed...", state, "completed")
        return state

    if state.revision_count == 0:
        state.revision_count = 1
        _progress(on_progress, "Stage 4 of 4: Revising...", state, "revising")
        try:
            state.final_brief = run_analyst(
                query,
                state.research_output,
                previous_draft=state.draft,
                feedback=state.critic_feedback,
            )
        except Exception as error:
            logger.error("Revision analyst call failed for query %r: %s", query, error)
            state.draft.missing_information.append(
                "The revision could not be completed."
            )
            state.final_brief = state.draft

    _mark_resumed_brief(state, resumed)
    _progress(on_progress, "Stage 4 of 4: Completed...", state, "completed")
    return state
