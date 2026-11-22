import logging
import json
from pathlib import Path
from typing import Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from market_research.agents.analyst import run_analyst
from market_research.agents.critic import run_critic
from market_research.agents.researcher import run_researcher
from market_research.agents.researcher import apply_loaded_research_rules
from market_research.config import ConfigError
from market_research.research.search import SearchQuotaError
from market_research.schemas import (
    CriticFeedback,
    ResearchBrief,
    ResearcherOutput,
    ResearchMode,
    WorkflowState,
    WorkflowStatus,
)
from market_research.services.llm import LLMQuotaError


logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str], None]
class PipelineGraphState(TypedDict, total=False):
    query: str
    research_output: ResearcherOutput | None
    draft: ResearchBrief | None
    critic_feedback: CriticFeedback | None
    final_brief: ResearchBrief | None
    revision_count: int
    status: WorkflowStatus
    error: str | None
    on_progress: ProgressCallback | None
    research_mode: ResearchMode


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


def apply_one_revision(
    revision_count: int,
    draft: ResearchBrief,
    revised_brief: ResearchBrief | None = None,
) -> tuple[ResearchBrief, int]:
    if revision_count >= 1:
        return draft, revision_count
    if revised_brief is None:
        return draft, revision_count
    return revised_brief, 1


def run_pipeline(
    query: str,
    on_progress: ProgressCallback | None = None,
    research_path: str | Path | None = None,
) -> WorkflowState:
    """Run the market research workflow through a LangGraph state graph."""
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    resumed = research_path is not None
    research_mode: ResearchMode = "resumed" if resumed else "live"

    def emit(callback: ProgressCallback | None, message: str) -> None:
        if callback:
            callback(message)

    def research_node(state: PipelineGraphState) -> PipelineGraphState:
        callback = state.get("on_progress")
        if research_path:
            try:
                path = Path(research_path)
                saved_text = path.read_text(encoding="utf-8")
                saved_payload = json.loads(saved_text)
                if isinstance(saved_payload, dict) and "research_output" in saved_payload:
                    saved_payload = saved_payload["research_output"]
                research_output = ResearcherOutput.model_validate(saved_payload)
                research_output = apply_loaded_research_rules(research_output)
                emit(callback, f"MODE: RESUMED from saved research ({path})")
                return {
                    "research_output": research_output,
                    "status": "pending",
                    "research_mode": "resumed",
                }
            except Exception as error:
                logger.error("Could not load saved research from %s", research_path)
                return {
                    "research_output": None,
                    "status": "failed",
                    "error": "The saved research file could not be loaded or was invalid.",
                }

        emit(callback, "Stage 1 of 4: Researcher...")
        try:
            research_output = run_researcher(query, callback)
            try:
                output_dir = Path.cwd() / "outputs"
                output_dir.mkdir(parents=True, exist_ok=True)
                saved_output = {
                    "research_mode": "live",
                    "research_output": research_output.model_dump(mode="json"),
                }
                (output_dir / "last_research.json").write_text(
                    json.dumps(saved_output, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as error:
                logger.error("Could not save last research output: %s", type(error).__name__)
            if not research_output.findings:
                return {
                    "research_output": research_output,
                    "status": "failed",
                    "error": (
                        "No evidence could be collected, so no brief was written. "
                        "Check your search provider and API keys."
                    ),
                }
            return {
                "research_output": research_output,
                "status": "researching",
                "research_mode": "live",
            }
        except Exception as error:
            logger.error("Researcher failed: %s", type(error).__name__)
            if isinstance(error, SearchQuotaError):
                return {
                    "research_output": None,
                    "status": "failed",
                    "error": (
                        "The search provider's credits or quota are used up. "
                        "Check your search key and plan, then try again."
                    ),
                }
            if isinstance(error, LLMQuotaError):
                return {
                    "research_output": None,
                    "status": "failed",
                    "error": (
                        "The language-model provider's quota is used up. "
                        "Switch provider in .env or try again later."
                    ),
                }
            return {
                "research_output": None,
                "status": "failed",
                "error": _failure_message(error),
            }

    def research_route(state: PipelineGraphState) -> Literal["analyst", "end"]:
        if state.get("status") == "failed":
            return "end"
        research_output = state.get("research_output")
        if research_output is None or not research_output.findings:
            return "end"
        return "analyst"

    def analyst_node(state: PipelineGraphState) -> PipelineGraphState:
        callback = state.get("on_progress")
        emit(callback, "Stage 2 of 4: Analyst...")
        research_output = state.get("research_output")
        if research_output is None:
            return {
                "status": "failed",
                "error": "The workflow could not continue without research output.",
            }
        try:
            draft = run_analyst(query, research_output)
            return {
                "draft": draft,
                "status": "analysing",
            }
        except Exception as error:
            logger.error("First analyst call failed: %s", type(error).__name__)
            return {
                "draft": None,
                "status": "failed",
                "error": _failure_message(error),
            }

    def analyst_route(state: PipelineGraphState) -> Literal["critic", "end"]:
        return "end" if state.get("status") == "failed" else "critic"

    def critic_node(state: PipelineGraphState) -> PipelineGraphState:
        callback = state.get("on_progress")
        emit(callback, "Stage 3 of 4: Critic...")
        research_output = state.get("research_output")
        draft = state.get("draft")
        if research_output is None or draft is None:
            return {
                "status": "failed",
                "error": "The workflow could not continue without a draft.",
            }
        try:
            critic_feedback = run_critic(query, research_output, draft)
            return {
                "critic_feedback": critic_feedback,
                "status": "reviewing",
            }
        except Exception as error:
            logger.error("Critic failed: %s", type(error).__name__)
            draft.missing_information.append(
                "The automated review could not be completed."
            )
            return {
                "critic_feedback": None,
                "draft": draft,
                "final_brief": draft,
                "status": "completed",
                "revision_count": 0,
            }

    def critic_route(state: PipelineGraphState) -> Literal["completed", "revision"]:
        feedback = state.get("critic_feedback")
        if feedback is not None and not feedback.approved:
            return "revision"
        return "completed"

    def completed_node(state: PipelineGraphState) -> PipelineGraphState:
        callback = state.get("on_progress")
        emit(callback, "Stage 4 of 4: Completed...")
        draft = state.get("draft")
        feedback = state.get("critic_feedback")
        final_brief = state.get("final_brief")

        if final_brief is None and feedback is not None and feedback.approved:
            final_brief = draft
        if resumed and final_brief:
            final_brief.missing_information = [
                "This brief was produced from a saved research dataset, not from a new live search.",
                *[
                    item
                    for item in final_brief.missing_information
                    if item != "This brief was produced from a saved research dataset, not from a new live search."
                ],
            ]
        return {
            "final_brief": final_brief,
            "status": "completed",
        }

    def revision_node(state: PipelineGraphState) -> PipelineGraphState:
        callback = state.get("on_progress")
        emit(callback, "Stage 4 of 4: Revising...")
        research_output = state.get("research_output")
        draft = state.get("draft")
        feedback = state.get("critic_feedback")
        revision_count = state.get("revision_count", 0)
        if research_output is None or draft is None or feedback is None:
            return {
                "status": "failed",
                "error": "The workflow could not complete the revision.",
            }
        if revision_count >= 1:
            return {
                "final_brief": draft,
                "revision_count": 1,
                "research_mode": research_mode,
            }
        try:
            final_brief = run_analyst(
                query,
                research_output,
                previous_draft=draft,
                feedback=feedback,
            )
            final_brief, revision_count = apply_one_revision(
                revision_count,
                draft,
                final_brief,
            )
        except Exception as error:
            logger.error("Revision analyst call failed: %s", type(error).__name__)
            draft.missing_information.append(
                "The revision could not be completed."
            )
            final_brief = draft
            revision_count = 1
        return {
            "final_brief": final_brief,
            "revision_count": revision_count,
            "research_mode": research_mode,
        }

    graph = StateGraph(PipelineGraphState)
    graph.add_node("research", research_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("critic", critic_node)
    graph.add_node("revision", revision_node)
    graph.add_node("completed", completed_node)
    graph.add_edge(START, "research")
    graph.add_conditional_edges("research", research_route, {
        "analyst": "analyst",
        "end": END,
    })
    graph.add_conditional_edges("analyst", analyst_route, {
        "critic": "critic",
        "end": END,
    })
    graph.add_conditional_edges("critic", critic_route, {
        "completed": "completed",
        "revision": "revision",
    })
    graph.add_edge("revision", "completed")
    graph.add_edge("completed", END)

    initial: PipelineGraphState = {
        "query": query,
        "research_output": None,
        "draft": None,
        "critic_feedback": None,
        "final_brief": None,
        "revision_count": 0,
        "status": "pending",
        "error": None,
        "on_progress": on_progress,
        "research_mode": research_mode,
    }
    final_state = graph.compile().invoke(initial)
    workflow_status = final_state.get("status", "failed")
    return WorkflowState(
        query=query,
        research_output=final_state.get("research_output"),
        draft=final_state.get("draft"),
        critic_feedback=final_state.get("critic_feedback"),
        final_brief=final_state.get("final_brief"),
        revision_count=final_state.get("revision_count", 0),
        status=workflow_status,
        error=final_state.get("error"),
        research_mode=final_state.get("research_mode", research_mode),
    )
