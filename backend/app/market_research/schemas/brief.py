from pydantic import Field

from .base import StrictModel
from .research import ResearcherOutput


class BriefCompetitor(StrictModel):
    name: str
    description: str
    evidence_finding_ids: list[str] = Field(min_length=1)


class KeyObservation(StrictModel):
    text: str
    evidence_finding_ids: list[str] = Field(min_length=1)


class ResearchBrief(StrictModel):
    subject: str
    executive_summary: str
    competitors: list[BriefCompetitor] = Field(default_factory=list, max_length=2)
    key_observations: list[KeyObservation] = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


class CriticFeedback(StrictModel):
    approved: bool
    overall_assessment: str
    issues: list[str] = Field(default_factory=list)
    revision_instructions: list[str] = Field(default_factory=list)

    def model_post_init(self, __context) -> None:
        if not self.approved and not self.revision_instructions:
            raise ValueError(
                "approved=False requires at least one revision instruction"
            )


def check_brief_references(
    brief: ResearchBrief, research: ResearcherOutput
) -> list[str]:
    """Deterministic grounding check the Orchestrator runs after the Analyst.
    Returns a list of problems; empty list means every citation resolves."""
    finding_ids = {f.finding_id for f in research.findings}
    source_ids = {s.source_id for s in research.sources}
    problems: list[str] = []

    for c in brief.competitors:
        bad = set(c.evidence_finding_ids) - finding_ids
        if bad:
            problems.append(f"Competitor '{c.name}' cites unknown findings {sorted(bad)}")

    for i, o in enumerate(brief.key_observations, start=1):
        bad = set(o.evidence_finding_ids) - finding_ids
        if bad:
            problems.append(f"Observation {i} cites unknown findings {sorted(bad)}")

    bad = set(brief.source_ids) - source_ids
    if bad:
        problems.append(f"Brief lists unknown sources {sorted(bad)}")

    return problems