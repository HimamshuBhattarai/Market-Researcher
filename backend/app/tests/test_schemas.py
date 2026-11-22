import pytest
from pydantic import ValidationError

from market_research.schemas import (
    CriticFeedback,
    KeyObservation,
    ResearchBrief,
    ResearchFinding,
    ResearchPlan,
    ResearchQuestion,
    ResearcherOutput,
    Source,
)


def make_research_output():
    plan = ResearchPlan(
        subject="Food delivery",
        domain="Food delivery",
        geography="Nepal",
        research_dimensions=["Competition"],
        research_questions=[
            ResearchQuestion(
                question_id="q1",
                question="Which companies operate in Nepal?",
                research_dimension="Competition",
            )
        ],
    )
    source = Source(
        source_id="S1",
        title="Market report",
        url="https://example.com/report",
        publisher="Example Publisher",
        accessed_date="2026-11-22",
        source_type="article",
    )
    finding = ResearchFinding(
        finding_id="F1",
        research_question_id="q1",
        claim="Pathao operates in Nepal.",
        evidence="Pathao provides food delivery in Nepal.",
        source_ids=["S1"],
        confidence="high",
        information_type="fact",
        scope="target_geography",
    )
    return ResearcherOutput(plan=plan, findings=[finding], sources=[source])


def test_research_output_accepts_valid_data():
    output = make_research_output()
    assert output.findings[0].finding_id == "F1"
    assert output.sources[0].source_id == "S1"


def test_research_output_rejects_unknown_fields():
    payload = make_research_output().model_dump()
    payload["unexpected"] = True

    with pytest.raises(ValidationError):
        ResearcherOutput.model_validate(payload)


def test_research_output_rejects_unknown_source_reference():
    payload = make_research_output().model_dump()
    payload["findings"][0]["source_ids"] = ["S999"]

    with pytest.raises(ValidationError):
        ResearcherOutput.model_validate(payload)


def test_rejected_brief_requires_revision_instructions():
    with pytest.raises(ValueError):
        CriticFeedback(
            approved=False,
            overall_assessment="Problems found.",
            issues=["Unsupported claim"],
        )


def test_brief_accepts_up_to_two_competitors():
    brief = ResearchBrief(
        subject="Food delivery",
        executive_summary="The market is active.",
        competitors=[],
        key_observations=[
            KeyObservation(
                text="The market is active.",
                evidence_finding_ids=["F1"],
            )
        ],
    )
    assert brief.competitors == []
