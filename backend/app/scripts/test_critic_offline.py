import sys
import os
import tempfile
import types
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from market_research.agents.critic import (
    _attribution_issues,
    _code_checks,
    _scope_and_hedge_checks,
)
from market_research.agents.analyst import (
    _capitalise_subject,
    _compute_missing,
    _ensure_comparability_caveat,
    _fallback_summary,
    _is_market_size_finding,
    _is_market_size_figure,
    _market_size_sources,
    _summary_covers_brief,
    _attributed_reference_text,
    _QUANTITATIVE_FIGURE,
)
from market_research.agents.researcher import (
    _SourceRegistry,
    _low_trust,
    apply_loaded_research_rules,
)
from market_research.schemas import (
    KeyObservation,
    ResearchBrief,
    ResearcherOutput,
    Source,
)
import market_research.services.llm as llm
import market_research.config as config
import market_research.agents.orchestrator as orchestrator
import market_research.agents.researcher as researcher_agent


def issues_for(brief, research):
    return (
        _code_checks(brief, research)
        + _scope_and_hedge_checks(brief, research)
    )


def make_brief(observations):
    return ResearchBrief(
        subject="Food delivery market in India",
        executive_summary="The India food delivery market has several cited figures.",
        competitors=[],
        key_observations=[
            KeyObservation(
                text=text,
                evidence_finding_ids=finding_ids,
            )
            for text, finding_ids in observations
        ],
        source_ids=[],
        missing_information=[],
    )


def run_case(label, condition):
    print(("PASS" if condition else "FAIL") + ": " + label)
    return condition


indian_path = APP_DIR / "scripts" / "outputs" / "indian_market.json"
fallback_path = APP_DIR / "scripts" / "outputs" / "researcher_output.json"
research_path = indian_path if indian_path.exists() else fallback_path
research = ResearcherOutput.model_validate_json(
    research_path.read_text(encoding="utf-8")
)

clean = make_brief([
    ("Meal delivery is one market segment.", ["F1"]),
    ("Meal delivery and grocery delivery are two solutions.", ["F1"]),
])

unknown = make_brief([
    ("The market stands at 71.76 billion.", ["F1"]),
    ("Meal delivery and grocery delivery are two solutions.", ["F1"]),
])

leaked_id = make_brief([
    ("Meal delivery is one market segment. (F2)", ["F1"]),
    ("Meal delivery and grocery delivery are two solutions.", ["F1"]),
])

base_finding = research.findings[0]
broader = base_finding.model_copy(
    update={
        "finding_id": "FB",
        "claim": "Global food delivery revenue reached 55.58 billion.",
        "evidence": "The global food delivery market reached 55.58 billion.",
        "scope": "broader_context",
        "confidence": "high",
    }
)
broader_research = research.model_copy(
    update={"findings": [broader]}
)

scope_plain = make_brief([
    ("The global market reached 55.58 billion.", ["FB"]),
])
scope_labelled = make_brief([
    (f"Not specific to {research.plan.geography or research.plan.subject}: The global market reached 55.58 billion.", ["FB"]),
])

low = base_finding.model_copy(
    update={
        "finding_id": "FL",
        "claim": "A social post reports 55.58 billion.",
        "evidence": "A social post reports 55.58 billion.",
        "scope": "target_geography",
        "confidence": "low",
    }
)
low_research = research.model_copy(
    update={"findings": [low]}
)
low_plain = make_brief([
    ("The market reached 55.58 billion.", ["FL"]),
])
low_hedged = make_brief([
    ("According to a social post, the market reached 55.58 billion.", ["FL"]),
])

passed = True

passed &= run_case(
    "clean brief: zero issues",
    len(issues_for(clean, research)) == 0,
)

unknown_issues = issues_for(unknown, research)
passed &= run_case(
    "unknown figure 71.76: exactly one issue",
    len(unknown_issues) == 1 and any("71.76" in issue for issue in unknown_issues),
)

id_issues = issues_for(leaked_id, research)
passed &= run_case(
    "leaked ID F2: exactly one issue",
    len(id_issues) == 1
    and any("finding IDs" in issue for issue in id_issues),
)

passed &= run_case(
    "broader context without label: one issue",
    len(issues_for(scope_plain, broader_research)) == 1,
)
passed &= run_case(
    "broader context with label: no issue",
    len(issues_for(scope_labelled, broader_research)) == 0,
)

passed &= run_case(
    "low-confidence plain fact: one issue",
    len(issues_for(low_plain, low_research)) == 1,
)
passed &= run_case(
    "low-confidence hedged with according: no issue",
    len(issues_for(low_hedged, low_research)) == 0,
)

target_finding = next(
    (finding for finding in research.findings if finding.scope == "target_geography"),
    base_finding,
)
broader_finding = next(
    (finding for finding in research.findings if finding.scope == "broader_context"),
    broader,
)
mixed_research = research.model_copy(
    update={"findings": [target_finding, broader_finding]}
)
mixed_plain = make_brief([
    ("The market reached a cited figure.", [
        target_finding.finding_id,
        broader_finding.finding_id,
    ]),
])
mixed_labelled = make_brief([
    (f"Not specific to {research.plan.geography or research.plan.subject}: The global market reached a cited figure.", [
        target_finding.finding_id,
        broader_finding.finding_id,
    ]),
])

passed &= run_case(
    "mixed-scope observation without label: one issue",
    len(issues_for(mixed_plain, mixed_research)) >= 1,
)
passed &= run_case(
    "mixed-scope observation with label: no issue",
    len(issues_for(mixed_labelled, mixed_research)) == 0,
)

unknown_publisher = make_brief([
    ("Unknownpublisher.io gives a market figure of 55.58 billion.", [target_finding.finding_id]),
])
target_sources = {
    source.source_id: source
    for source in research.sources
}
target_publishers = [
    target_sources[source_id].publisher
    for source_id in target_finding.source_ids
    if source_id in target_sources and target_sources[source_id].publisher
]
cited_name = target_publishers[0] if target_publishers else "statista.com"
cited_publisher = make_brief([
    (f"{cited_name} gives a market figure.", [target_finding.finding_id]),
])

passed &= run_case(
    "unverified website publisher: one issue",
    len(issues_for(unknown_publisher, research)) >= 1,
)
passed &= run_case(
    "cited website publisher: no issue",
    len(issues_for(cited_publisher, research)) == 0,
)

estimate_data = target_finding.model_dump()
estimate_data["confidence"] = "estimate"
estimate_finding = type(target_finding).model_validate(estimate_data)
passed &= run_case(
    "confidence estimate loads as low",
    estimate_finding.confidence == "low",
)

list_finding = target_finding.model_copy(
    update={
        "finding_id": "FMT",
        "claim": "(1) Meal Delivery and (2) Grocery Delivery",
        "evidence": "(1) Meal Delivery and (2) Grocery Delivery",
    }
)
quantitative_finding = target_finding.model_copy(
    update={
        "finding_id": "FMQ",
        "claim": "The market is projected to reach US$61.76bn in 2026.",
        "evidence": "The market is projected to reach US$61.76bn in 2026.",
    }
)
tam_plan = research.plan.model_copy(
    update={"required_metrics": ["Total addressable market (TAM)"]}
)

list_research = research.model_copy(
    update={"plan": tam_plan, "findings": [list_finding]}
)
quantitative_research = research.model_copy(
    update={"plan": tam_plan, "findings": [quantitative_finding]}
)

passed &= run_case(
    "list numbering does not cover TAM",
    "Total addressable market (TAM)" in _compute_missing(list_research),
)
passed &= run_case(
    "US$ market figure covers TAM",
    "Total addressable market (TAM)" not in _compute_missing(quantitative_research),
)

rupee_finding = target_finding.model_copy(
    update={
        "finding_id": "F-RUPEE",
        "claim": "Delivery cost per order, usually " + chr(8377) + "40" + chr(8211) + chr(8377) + "60 in urban India",
        "evidence": "Delivery cost per order, usually " + chr(8377) + "40" + chr(8211) + chr(8377) + "60 in urban India",
    }
)
rupee_plan = research.plan.model_copy(
    update={"required_metrics": ["Delivery cost per order"]}
)
rupee_research = research.model_copy(
    update={"plan": rupee_plan, "findings": [rupee_finding]}
)
passed &= run_case(
    "rupee range counts for delivery cost per order",
    bool(_QUANTITATIVE_FIGURE.search(chr(8377) + "40" + chr(8211) + chr(8377) + "60"))
    and "Delivery cost per order" not in _compute_missing(rupee_research),
)

commission_finding = target_finding.model_copy(
    update={
        "finding_id": "F-COMMISSION",
        "claim": "Commission per order is 25%.",
        "evidence": "Commission per order is 25%.",
    }
)
passed &= run_case(
    "commission finding does not count as market size",
    not _is_market_size_finding(
        f"{commission_finding.claim} {commission_finding.evidence}"
    ),
)

blog_registry = _SourceRegistry()
blog_registry.sources["S-BLOG"] = Source(
    source_id="S-BLOG",
    title="TechDino blog",
    url="https://techdino.in/blog/food-delivery",
    publisher="techdino.in",
    accessed_date="2026-11-22",
    source_type="web page",
)
passed &= run_case(
    "blog URL caps confidence at low",
    _low_trust(["S-BLOG"], blog_registry),
)

size_publishers = ["Market Research Future", "Nexdigm"]
base_summary = "The market has figures from separate studies."
appended = _ensure_comparability_caveat(base_summary, size_publishers)
present = _ensure_comparability_caveat(appended, size_publishers)
passed &= run_case(
    "comparability sentence appended once",
    appended.startswith(base_summary)
    and "sources" in appended.lower()
    and "marketresearchfuture" not in appended.lower()
    and present == appended,
)

passed &= run_case(
    "brief subject is capitalised",
    _capitalise_subject("food delivery market in India")
    == "Food delivery market in India",
)

market_finding_a = target_finding.model_copy(
    update={
        "finding_id": "F-MA",
        "claim": "The market grew from USD 55.58 Billion in 2025.",
        "evidence": "The market grew from USD 55.58 Billion in 2025.",
        "source_ids": ["S-MA"],
    }
)
market_finding_b = target_finding.model_copy(
    update={
        "finding_id": "F-MB",
        "claim": "The market attained a value of nearly US$ 5.30 Billion in 2022.",
        "evidence": "The market attained a value of nearly US$ 5.30 Billion in 2022.",
        "source_ids": ["S-MB"],
    }
)
comparability_research = research.model_copy(
    update={
        "findings": [market_finding_a, market_finding_b],
        "sources": [
            Source(
                source_id="S-MA",
                title="IMARC",
                url="https://www.imarcgroup.com/market",
                publisher="imarcgroup.com",
                accessed_date="2026-11-22",
                source_type="web page",
            ),
            Source(
                source_id="S-MB",
                title="Yahoo",
                url="https://finance.yahoo.com/market",
                publisher="finance.yahoo.com",
                accessed_date="2026-11-22",
                source_type="web page",
            ),
        ],
    }
)

passed &= run_case(
    "market figures from two publishers trigger comparability",
    len(_market_size_sources(comparability_research)) >= 2
    and _is_market_size_figure(market_finding_a.claim)
    and _is_market_size_figure(market_finding_b.claim),
)

commission_finding = target_finding.model_copy(
    update={
        "finding_id": "F-COMMISSION-2",
        "claim": "Commission per order is USD 5 Billion.",
        "evidence": "Commission per order is USD 5 Billion.",
    }
)
passed &= run_case(
    "commission market-size-like figure is excluded",
    not _is_market_size_figure(commission_finding.claim),
)

caveat_publishers = ["imarcgroup.com", "finance.yahoo.com"]
caveat_missing = _ensure_comparability_caveat(
    "The market has separate figures.",
    caveat_publishers,
)
caveat_present = _ensure_comparability_caveat(
    caveat_missing,
    caveat_publishers,
)
passed &= run_case(
    "caveat appended once",
    caveat_missing != "The market has separate figures."
    and caveat_present == caveat_missing,
)

competitor = types.SimpleNamespace(name="Ownly")
regulation_observation = types.SimpleNamespace(
    text="Food safety regulation is a compliance challenge."
)
passed &= run_case(
    "competitor summary omission triggers fallback",
    not _summary_covers_brief(
        "The market has figures.",
        [competitor],
        [regulation_observation],
    ),
)

blog_registry = _SourceRegistry()
blog_registry.sources["S-BLOGS"] = Source(
    source_id="S-BLOGS",
    title="Cyril Amarchand blogs",
    url="https://corporate.cyrilamarchandblogs.com/article",
    publisher="corporate.cyrilamarchandblogs.com",
    accessed_date="2026-11-22",
    source_type="web page",
)
passed &= run_case(
    "blogs host caps confidence at low",
    _low_trust(["S-BLOGS"], blog_registry),
)

advisory_finding = target_finding.model_copy(
    update={
        "finding_id": "F-ADVISORY",
        "claim": "The Advisory places significant responsibility.",
        "evidence": "The Advisory places significant responsibility.",
        "source_ids": ["S6"],
    }
)
advisory_research = research.model_copy(
    update={"findings": [advisory_finding]}
)
advisory_text = _attributed_reference_text(
    "The Advisory places significant responsibility.",
    ["F-ADVISORY"],
    advisory_research,
)
advisory_brief = make_brief([
    (
        "The Advisory places significant responsibility.",
        ["F-ADVISORY"],
    ),
])
passed &= run_case(
    "unnamed advisory gets prefix and critic issue",
    advisory_text.startswith("According to ")
    and len(_scope_and_hedge_checks(advisory_brief, advisory_research)) >= 1,
)

passed &= run_case(
    "subject receives geography",
    _capitalise_subject("Food delivery market", "India")
    == "Food delivery market in India",
)

attr_source_a = Source(
    source_id="S-ATTR-A",
    title="IMARC",
    url="https://imarcgroup.com/report",
    publisher="imarcgroup.com",
    accessed_date="2026-11-22",
    source_type="web page",
)
attr_source_b = Source(
    source_id="S-ATTR-B",
    title="Yahoo",
    url="https://finance.yahoo.com/report",
    publisher="finance.yahoo.com",
    accessed_date="2026-11-22",
    source_type="web page",
)
attr_source_c = Source(
    source_id="S-ATTR-C",
    title="Cyril Amarchand",
    url="https://corporate.cyrilamarchandblogs.com/article",
    publisher="corporate.cyrilamarchandblogs.com",
    accessed_date="2026-11-22",
    source_type="web page",
)
attr_source_d = Source(
    source_id="S-ATTR-D",
    title="Acuity",
    url="https://acuitylaw.co.in/article",
    publisher="acuitylaw.co.in",
    accessed_date="2026-11-22",
    source_type="web page",
)
attr_finding_a = target_finding.model_copy(
    update={
        "finding_id": "F-ATTR-A",
        "claim": "The market was valued at USD 55.58 Billion.",
        "evidence": "The market was valued at USD 55.58 Billion.",
        "source_ids": ["S-ATTR-A"],
    }
)
attr_finding_b = target_finding.model_copy(
    update={
        "finding_id": "F-ATTR-B",
        "claim": "The market reached USD 67.90 Billion.",
        "evidence": "The market reached USD 67.90 Billion.",
        "source_ids": ["S-ATTR-B"],
    }
)
attr_finding_c = target_finding.model_copy(
    update={
        "finding_id": "F-ATTR-C",
        "claim": "Restaurants lack FSSAI registration.",
        "evidence": "Restaurants lack FSSAI registration.",
        "source_ids": ["S-ATTR-C"],
    }
)
attr_research = research.model_copy(
    update={
        "findings": [attr_finding_a, attr_finding_b, attr_finding_c],
        "sources": [
            attr_source_a,
            attr_source_b,
            attr_source_c,
            attr_source_d,
        ],
    }
)

def attr_brief(summary):
    return ResearchBrief(
        subject="Food delivery market",
        executive_summary=summary,
        competitors=[],
        key_observations=[
            KeyObservation(
                text="A valid market observation.",
                evidence_finding_ids=["F-ATTR-A"],
            )
        ],
        source_ids=["S-ATTR-A"],
    )

wrong_attr = attr_brief(
    "According to finance.yahoo.com, the market was valued at "
    "USD 55.58 Billion."
)
right_attr = attr_brief(
    "According to imarcgroup.com, the market was valued at "
    "USD 55.58 Billion."
)
mixed_attr = attr_brief(
    "IMARC gave USD 55.58 Billion and finance.yahoo.com gave "
    "USD 67.90 Billion."
)

passed &= run_case(
    "wrong publisher attribution gives one issue",
    len(_attribution_issues(wrong_attr, attr_research)) == 1,
)
passed &= run_case(
    "correct publisher attribution gives no issue",
    len(_attribution_issues(right_attr, attr_research)) == 0,
)
passed &= run_case(
    "mixed-publisher summary figures give one issue",
    len(_attribution_issues(mixed_attr, attr_research)) >= 1,
)

acuity_attr = attr_brief(
    "According to acuitylaw.co.in, restaurants lack FSSAI registration."
)
passed &= run_case(
    "wrong acuity attribution gives one issue",
    len(_attribution_issues(acuity_attr, attr_research)) == 1,
)

fallback_observation = types.SimpleNamespace(
    text="According to imarcgroup.com, the market was valued at USD 55.58 Billion."
)
fallback_summary = _fallback_summary(
    attr_research,
    [],
    [fallback_observation],
)
fallback_brief = attr_brief(fallback_summary)
passed &= run_case(
    "failing summary triggers passing fallback",
    len(_attribution_issues(wrong_attr, attr_research)) >= 1
    and len(_attribution_issues(fallback_brief, attr_research)) == 0,
)

loaded_finding = attr_finding_c.model_copy(
    update={
        "finding_id": "F-LOADED-BLOG",
        "confidence": "medium",
        "scope": "target_geography",
        "source_ids": ["S-ATTR-C"],
    }
)
loaded_research = attr_research.model_copy(
    update={"findings": [loaded_finding]}
)
normalized_research = apply_loaded_research_rules(loaded_research)
passed &= run_case(
    "loaded blogs finding becomes low confidence",
    normalized_research.findings[0].confidence == "low",
)

class FakeStatusError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.status_code = 400
        self.message = message


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


settings = types.SimpleNamespace(
    llm_provider="groq",
    llm_api_key="test",
    llm_model="openai/gpt-oss-20b",
    llm_base_url="https://example.invalid",
    llm_json_mode="on",
)

original_status_error = llm.openai.APIStatusError
original_get_settings = llm.get_settings
original_openai_client = llm._openai_client
llm.openai.APIStatusError = FakeStatusError
llm.get_settings = lambda: settings

try:
    llm._json_mode_disabled = False
    llm._consecutive_json_400 = 0
    first_specs = []
    first_results = [
        FakeStatusError("Failed to generate JSON"),
        FakeResponse('{"answer":"ok"}'),
        FakeResponse('{"answer":"next"}'),
    ]

    class FirstClient:
        def __init__(self):
            self.calls = 0
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self.create)
            )
        def create(self, **kwargs):
            first_specs.append(kwargs)
            result = first_results[self.calls]
            self.calls += 1
            if isinstance(result, Exception):
                raise result
            return result

    first_client = FirstClient()
    llm._openai_client = lambda: first_client

    first = llm._complete_with_openai_compatible([
        {"role": "user", "content": "Return JSON"}
    ])
    llm._complete_with_openai_compatible([
        {"role": "user", "content": "Return JSON again"}
    ])
    passed &= run_case(
        "one JSON 400 retries once without JSON mode and remains enabled",
        first == '{"answer":"ok"}'
        and len(first_specs) == 3
        and "response_format" in first_specs[0]
        and "response_format" not in first_specs[1]
        and "response_format" in first_specs[2]
        and not llm._json_mode_disabled,
    )

    llm._json_mode_disabled = False
    llm._consecutive_json_400 = 0
    second_specs = []
    second_results = [
        FakeStatusError("Failed to validate JSON"),
        FakeStatusError("Failed to generate JSON"),
        FakeStatusError("Failed to validate JSON"),
        FakeStatusError("failed_generation"),
    ]

    class SecondClient:
        def __init__(self):
            self.calls = 0
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self.create)
            )
        def create(self, **kwargs):
            second_specs.append(kwargs)
            result = second_results[self.calls]
            self.calls += 1
            if isinstance(result, Exception):
                raise result
            return result

    llm._openai_client = lambda: SecondClient()
    second_failed = False
    try:
        llm._complete_with_openai_compatible([
            {"role": "user", "content": "First call"}
        ])
    except llm.LLMError:
        second_failed = True
    try:
        llm._complete_with_openai_compatible([
            {"role": "user", "content": "Second call"}
        ])
    except llm.LLMError:
        second_failed = True

    passed &= run_case(
        "two JSON failures in a row switch JSON mode off",
        second_failed
        and llm._json_mode_disabled
        and len(second_specs) == 4,
    )

    llm._json_mode_disabled = False
    llm._consecutive_json_400 = 0
    settings.llm_model = "openai/gpt-oss-120b"
    settings.llm_json_mode = "auto"
    gpt_specs = []

    class GptClient:
        def __init__(self):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self.create)
            )
        def create(self, **kwargs):
            gpt_specs.append(kwargs)
            return FakeResponse('{"answer":"ok"}')

    llm._openai_client = lambda: GptClient()
    llm._complete_with_openai_compatible([
        {"role": "user", "content": "Return JSON"}
    ])
    passed &= run_case(
        "GPT OSS auto starts without JSON mode",
        "response_format" not in gpt_specs[0],
    )
finally:
    llm.openai.APIStatusError = original_status_error
    llm.get_settings = original_get_settings
    llm._openai_client = original_openai_client

original_researcher = orchestrator.run_researcher
original_analyst = orchestrator.run_analyst
original_cwd = os.getcwd()
try:
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        orchestrator.run_researcher = lambda query, callback=None: research

        def failing_analyst(*args, **kwargs):
            raise RuntimeError("simulated analyst failure")

        orchestrator.run_analyst = failing_analyst
        saved_state = orchestrator.run_pipeline(
            "Analyze the food delivery market",
            on_progress=None,
        )
        saved_path = Path(temp_dir) / "outputs" / "last_research.json"
        passed &= run_case(
            "research is saved before analyst failure",
            saved_state.status == "failed"
            and saved_path.exists()
            and ResearcherOutput.model_validate_json(
                saved_path.read_text(encoding="utf-8")
            ).plan.subject == research.plan.subject,
        )
        os.chdir(original_cwd)
finally:
    os.chdir(original_cwd)
    orchestrator.run_researcher = original_researcher
    orchestrator.run_analyst = original_analyst

original_researcher = orchestrator.run_researcher
original_analyst = orchestrator.run_analyst
research_called = False
try:
    def unexpected_research(*args, **kwargs):
        global research_called
        research_called = True
        raise RuntimeError("research should be skipped")

    def failing_resumed_analyst(*args, **kwargs):
        raise RuntimeError("simulated resumed analyst failure")

    orchestrator.run_researcher = unexpected_research
    orchestrator.run_analyst = failing_resumed_analyst
    saved_state = orchestrator.run_pipeline(
        "Analyze the saved market",
        on_progress=None,
        research_path=research_path if "research_path" in globals() else (
            APP_DIR / "scripts" / "outputs" / "indian_market.json"
        ),
    )
    passed &= run_case(
        "saved research skips researcher",
        not research_called
        and saved_state.status == "failed",
    )
finally:
    orchestrator.run_researcher = original_researcher
    orchestrator.run_analyst = original_analyst

raise SystemExit(0 if passed else 1)
