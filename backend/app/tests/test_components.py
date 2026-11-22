from market_research.agents import analyst, critic, researcher
from market_research.research import planner
from market_research.schemas import (
    BriefCompetitor,
    KeyObservation,
    ResearchBrief,
)

from .test_schemas import make_research_output


def test_planner_maps_draft_to_research_plan(monkeypatch):
    draft = planner._PlanDraft(
        subject="Food delivery",
        domain="Food delivery",
        geography="Nepal",
        research_questions=[
            planner._QuestionDraft(
                question="Who competes?",
                research_dimension="Competition",
            )
        ],
        required_metrics=["Market growth rate"],
    )
    monkeypatch.setattr(
        planner,
        "generate_structured",
        lambda *_args, **_kwargs: draft,
    )

    plan = planner.make_plan("Analyze food delivery in Nepal")

    assert plan.research_questions[0].question_id == "q1"
    assert plan.research_dimensions == ["Competition"]


def test_researcher_reapplies_low_trust_rule():
    output = make_research_output()
    output.sources[0].publisher = "facebook.com"

    updated = researcher.apply_loaded_research_rules(output)

    assert updated.findings[0].confidence == "low"


def test_analyst_strip_ids():
    assert analyst._strip_ids("Pathao leads (F1, F2)") == "Pathao leads"


def test_critic_accepts_supported_source():
    output = make_research_output()
    brief = ResearchBrief(
        subject="Food delivery in Nepal",
        executive_summary="Pathao operates in Nepal.",
        competitors=[
            BriefCompetitor(
                name="Pathao",
                description="Pathao operates in Nepal.",
                evidence_finding_ids=["F1"],
            )
        ],
        key_observations=[
            KeyObservation(
                text="Pathao operates in Nepal.",
                evidence_finding_ids=["F1"],
            )
        ],
        source_ids=["S1"],
    )

    assert critic._code_checks(brief, output) == []
