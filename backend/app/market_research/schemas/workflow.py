from typing import Literal

from pydantic import ConfigDict, Field

from .base import StrictModel
from .brief import CriticFeedback, ResearchBrief
from .research import ResearcherOutput

WorkflowStatus = Literal[
    "pending",
    "researching",
    "analysing",
    "reviewing",
    "revising",
    "completed",
    "failed",
]

ResearchMode = Literal["live", "resumed"]


class WorkflowState(StrictModel):
    # validate_assignment makes the revision cap and status values hold
    # when the Orchestrator mutates the state, not just at construction.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    query: str = Field(min_length=1)
    research_output: ResearcherOutput | None = None
    draft: ResearchBrief | None = None
    critic_feedback: CriticFeedback | None = None
    revision_count: int = Field(default=0, ge=0, le=1)
    research_mode: ResearchMode = "live"
    final_brief: ResearchBrief | None = None
    status: WorkflowStatus = "pending"
    error: str | None = None