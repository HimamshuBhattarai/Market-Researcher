from typing import Literal

from pydantic import Field, field_validator, model_validator

from .base import StrictModel

Confidence = Literal["high", "medium", "low"]
InformationType = Literal["fact", "estimate", "interpretation"]


def _duplicates(ids: list[str]) -> list[str]:
    return sorted({i for i in ids if ids.count(i) > 1})


class ResearchQuestion(StrictModel):
    question_id: str
    question: str
    research_dimension: str
    relevant_metrics: list[str] = Field(default_factory=list)


class ResearchPlan(StrictModel):
    subject: str
    domain: str
    geography: str | None = None
    research_dimensions: list[str] = Field(default_factory=list)
    research_questions: list[ResearchQuestion] = Field(min_length=1)
    required_metrics: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_questions(self):
        dupes = _duplicates([q.question_id for q in self.research_questions])
        if dupes:
            raise ValueError(f"Duplicate question_ids: {dupes}")
        dims = set(self.research_dimensions)
        for q in self.research_questions:
            if q.research_dimension not in dims:
                raise ValueError(
                    f"{q.question_id} uses dimension '{q.research_dimension}' "
                    f"which is not in research_dimensions"
                )
        return self


class Source(StrictModel):
    """Built by your code from search/scrape results, never by the LLM."""

    source_id: str
    title: str
    url: str
    publisher: str | None = None
    publication_date: str | None = None
    accessed_date: str  # ISO date, set by code
    source_type: str
    
Scope = Literal["target_geography", "broader_context"]


class ExtractedFinding(StrictModel):
    """What the LLM fills in. Question and finding IDs are attached by code."""

    claim: str
    evidence: str  # short quote or close excerpt from the source
    source_ids: list[str] = Field(min_length=1)
    confidence: Confidence
    information_type: InformationType
    scope: Scope

    @field_validator("scope", mode="before")
    @classmethod
    def normalise_scope(cls, v):
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("target_geography", "broader_context"):
                return s
            if any(w in s for w in ("global", "world", "broader", "other", "region", "general")):
                return "broader_context"
            return "target_geography"
        return v


class ResearchFinding(ExtractedFinding):
    finding_id: str
    research_question_id: str

class ResearcherOutput(StrictModel):
    plan: ResearchPlan
    findings: list[ResearchFinding] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_references(self):
        source_ids = [s.source_id for s in self.sources]
        finding_ids = [f.finding_id for f in self.findings]

        for label, ids in (("source_id", source_ids), ("finding_id", finding_ids)):
            dupes = _duplicates(ids)
            if dupes:
                raise ValueError(f"Duplicate {label}s: {dupes}")

        known_questions = {q.question_id for q in self.plan.research_questions}
        known_sources = set(source_ids)

        for f in self.findings:
            if f.research_question_id not in known_questions:
                raise ValueError(
                    f"{f.finding_id} references unknown question "
                    f"{f.research_question_id}"
                )
            unknown = set(f.source_ids) - known_sources
            if unknown:
                raise ValueError(
                    f"{f.finding_id} cites unknown sources {sorted(unknown)}"
                )
        return self