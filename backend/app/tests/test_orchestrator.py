from market_research.agents import critic, orchestrator
from market_research.research.search import SearchQuotaError
from market_research.services.llm import LLMQuotaError
import pytest

from market_research.schemas import (
    BriefCompetitor,
    KeyObservation,
    KeyObservation,
    ResearchBrief,
)

from .test_schemas import make_research_output


def test_orchestrator_runs_langgraph_with_mocked_agents(monkeypatch):
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

    monkeypatch.setattr(
        orchestrator,
        "run_researcher",
        lambda _query, callback=None: output,
    )
    monkeypatch.setattr(
        orchestrator,
        "run_analyst",
        lambda _query, _research, previous_draft=None, feedback=None: brief,
    )
    monkeypatch.setattr(
        orchestrator,
        "run_critic",
        lambda _query, _research, _draft: critic.CriticFeedback(
            approved=True,
            overall_assessment="No material problems found.",
        ),
    )

    state = orchestrator.run_pipeline("Analyze food delivery in Nepal")

    assert state.status == "completed"
    assert state.final_brief is not None
    assert state.final_brief.subject == "Food delivery in Nepal"


def test_search_quota_stops_before_analyst(monkeypatch):
    analyst_calls = []
    monkeypatch.setattr(
        orchestrator,
        "run_researcher",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            SearchQuotaError("quota")
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "run_analyst",
        lambda *_args, **_kwargs: analyst_calls.append(True),
    )

    state = orchestrator.run_pipeline("Analyze food delivery")

    assert state.status == "failed"
    assert "credits or quota" in (state.error or "")
    assert analyst_calls == []


def test_language_quota_stops_before_analyst(monkeypatch):
    analyst_calls = []
    monkeypatch.setattr(
        orchestrator,
        "run_researcher",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            LLMQuotaError("daily limit")
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "run_analyst",
        lambda *_args, **_kwargs: analyst_calls.append(True),
    )

    state = orchestrator.run_pipeline("Analyze food delivery")

    assert state.status == "failed"
    assert "language-model provider" in (state.error or "")
    assert analyst_calls == []


def test_zero_findings_stop_before_analyst(monkeypatch):
    output = make_research_output()
    output = output.model_copy(update={"findings": [], "sources": []})
    analyst_calls = []
    monkeypatch.setattr(
        orchestrator,
        "run_researcher",
        lambda *_args, **_kwargs: output,
    )
    monkeypatch.setattr(
        orchestrator,
        "run_analyst",
        lambda *_args, **_kwargs: analyst_calls.append(True),
    )

    state = orchestrator.run_pipeline("Analyze food delivery")

    assert state.status == "failed"
    assert "No evidence could be collected" in (state.error or "")
    assert analyst_calls == []


def test_approved_review_has_zero_revision(monkeypatch):
    output = make_research_output()
    brief = ResearchBrief(
        subject="Food delivery",
        executive_summary="Pathao operates in Nepal.",
        competitors=[],
        key_observations=[
            KeyObservation(text="Evidence-backed observation.", evidence_finding_ids=["F1"])
        ],
    )
    critic_calls = []
    monkeypatch.setattr(
        orchestrator,
        "run_researcher",
        lambda *_args, **_kwargs: output,
    )
    monkeypatch.setattr(
        orchestrator,
        "run_analyst",
        lambda *_args, **_kwargs: brief,
    )
    monkeypatch.setattr(
        orchestrator,
        "run_critic",
        lambda *_args, **_kwargs: (
            critic_calls.append(True),
            critic.CriticFeedback(
                approved=True,
                overall_assessment="No problems.",
            ),
        )[1],
    )

    state = orchestrator.run_pipeline("Analyze food delivery")

    assert state.revision_count == 0
    assert len(critic_calls) == 1
    assert state.final_brief is not None


def test_not_approved_has_one_revision(monkeypatch):
    output = make_research_output()
    first_brief = ResearchBrief(
        subject="Food delivery",
        executive_summary="First draft.",
        competitors=[],
        key_observations=[
            KeyObservation(text="Evidence-backed observation.", evidence_finding_ids=["F1"])
        ],
    )
    revised_brief = ResearchBrief(
        subject="Food delivery",
        executive_summary="Revised draft.",
        competitors=[],
        key_observations=[
            KeyObservation(text="Evidence-backed observation.", evidence_finding_ids=["F1"])
        ],
    )
    analyst_calls = []
    critic_calls = []

    monkeypatch.setattr(
        orchestrator,
        "run_researcher",
        lambda *_args, **_kwargs: output,
    )

    def fake_analyst(*_args, **_kwargs):
        analyst_calls.append(True)
        return first_brief if len(analyst_calls) == 1 else revised_brief

    monkeypatch.setattr(orchestrator, "run_analyst", fake_analyst)
    monkeypatch.setattr(
        orchestrator,
        "run_critic",
        lambda *_args, **_kwargs: (
            critic_calls.append(True),
            critic.CriticFeedback(
                approved=False,
                overall_assessment="Problems found.",
                issues=["Unsupported claim"],
                revision_instructions=["Fix the unsupported claim"],
            ),
        )[1],
    )

    state = orchestrator.run_pipeline("Analyze food delivery")

    assert state.revision_count == 1
    assert len(critic_calls) == 1
    assert len(analyst_calls) == 2
    assert state.final_brief is revised_brief


def test_revision_node_does_nothing_at_count_one():
    draft = ResearchBrief(
        subject="Food delivery",
        executive_summary="Existing draft.",
        competitors=[],
        key_observations=[
            KeyObservation(text="Evidence-backed observation.", evidence_finding_ids=["F1"])
        ],
    )
    final_brief, count = orchestrator.apply_one_revision(
        1,
        draft,
        revised_brief=ResearchBrief(
            subject="Food delivery",
            executive_summary="Should not be used.",
            competitors=[],
            key_observations=[
                KeyObservation(text="Evidence-backed observation.", evidence_finding_ids=["F1"])
            ],
        ),
    )
    assert count == 1
    assert final_brief is draft


def test_live_and_resumed_research_modes(monkeypatch, tmp_path):
    output = make_research_output()
    brief = ResearchBrief(
        subject="Food delivery",
        executive_summary="Brief.",
        competitors=[],
        key_observations=[
            KeyObservation(text="Evidence-backed observation.", evidence_finding_ids=["F1"])
        ],
    )
    monkeypatch.setattr(
        orchestrator,
        "run_researcher",
        lambda *_args, **_kwargs: output,
    )
    monkeypatch.setattr(
        orchestrator,
        "run_analyst",
        lambda *_args, **_kwargs: brief,
    )
    monkeypatch.setattr(
        orchestrator,
        "run_critic",
        lambda *_args, **_kwargs: critic.CriticFeedback(
            approved=True,
            overall_assessment="No problems.",
        ),
    )

    live = orchestrator.run_pipeline("Analyze food delivery")
    assert live.research_mode == "live"

    saved = tmp_path / "research.json"
    saved.write_text(output.model_dump_json(), encoding="utf-8")
    resumed = orchestrator.run_pipeline(
        "Analyze food delivery",
        research_path=saved,
    )
    assert resumed.research_mode == "resumed"


def test_critic_failure_falls_back_to_draft(monkeypatch):
    output = make_research_output()
    brief = ResearchBrief(
        subject="Food delivery",
        executive_summary="First draft.",
        competitors=[],
        key_observations=[
            KeyObservation(text="Evidence-backed observation.", evidence_finding_ids=["F1"])
        ],
    )
    monkeypatch.setattr(orchestrator, "run_researcher", lambda *_args, **_kwargs: output)
    monkeypatch.setattr(orchestrator, "run_analyst", lambda *_args, **_kwargs: brief)

    def fail_critic(*_args, **_kwargs):
        raise RuntimeError("critic unavailable")

    monkeypatch.setattr(orchestrator, "run_critic", fail_critic)
    state = orchestrator.run_pipeline("Analyze food delivery")
    assert state.final_brief is brief
    assert "The automated review could not be completed." in brief.missing_information


def test_revision_failure_falls_back_to_draft(monkeypatch):
    output = make_research_output()
    brief = ResearchBrief(
        subject="Food delivery",
        executive_summary="First draft.",
        competitors=[],
        key_observations=[
            KeyObservation(text="Evidence-backed observation.", evidence_finding_ids=["F1"])
        ],
    )
    analyst_calls = []
    monkeypatch.setattr(orchestrator, "run_researcher", lambda *_args, **_kwargs: output)

    def fake_analyst(*_args, **_kwargs):
        analyst_calls.append(True)
        if len(analyst_calls) == 1:
            return brief
        raise RuntimeError("revision unavailable")

    monkeypatch.setattr(orchestrator, "run_analyst", fake_analyst)
    monkeypatch.setattr(
        orchestrator,
        "run_critic",
        lambda *_args, **_kwargs: critic.CriticFeedback(
            approved=False,
            overall_assessment="Problems found.",
            issues=["Unsupported claim"],
            revision_instructions=["Fix it"],
        ),
    )
    state = orchestrator.run_pipeline("Analyze food delivery")
    assert state.final_brief is brief
    assert "The revision could not be completed." in brief.missing_information
