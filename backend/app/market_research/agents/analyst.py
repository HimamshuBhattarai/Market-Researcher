import json
import re

from pydantic import Field

from market_research.schemas import (
    BriefCompetitor,
    CriticFeedback,
    KeyObservation,
    ResearchBrief,
    ResearcherOutput,
    check_brief_references,
)
from market_research.schemas.base import StrictModel
from market_research.agents.critic import HEDGE_WORDS, _attribution_issues
from market_research.services.llm import LLMError, generate_structured

SYSTEM = (
    "You are a market research analyst writing a short, evidence-backed brief. "
    "You only use the findings provided. You never use outside knowledge and "
    "never invent facts, figures, companies or sources."
)

MAX_OBSERVATIONS = 6
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")

METRIC_KEYWORDS = {
    "total addressable market": (
        "market size",
        "stands at",
        "valued",
        "market is projected",
        "market is expected",
        "revenue in the",
        "market revenue",
    ),
    "market growth rate": ("cagr", "growth rate", "growing at", "compound annual", "year-over-year", "yoy"),
    "number of active users": ("users", "consumers used", "active customers", "subscribers"),
    "average order value": ("average order value", "aov", "order value"),
    "order frequency": ("order frequency", "orders per", "times a month", "times per month", "frequency"),
    "customer acquisition cost": ("acquisition cost", "cac"),
    "customer lifetime value": ("lifetime value", "ltv"),
    "delivery cost per order": ("delivery cost", "cost per order"),
    "commission rate": ("commission",),
    "customer retention": ("retention", "repeat customers", "churn"),
}

MARKET_SIZE_EXCLUDED_TERMS = (
    "commission",
    "fee",
    "fees",
    "per order",
    "cost",
    "pay",
    "margin",
)


def _is_market_size_finding(text: str) -> bool:
    lowered = text.lower()
    if any(term in lowered for term in MARKET_SIZE_EXCLUDED_TERMS):
        return False
    return any(
        keyword in lowered
        for keyword in METRIC_KEYWORDS["total addressable market"]
    )


_MARKET_SIZE_AMOUNT = re.compile(
    r"(?:"
    r"(?:US\$|\$|USD|NPR|INR|Rs\.?|₹|₨|EUR|€|GBP|£|JPY|CNY|¥)"
    r"\s*\d+(?:[.,]\d)*\s*(?:billion|million|trillion|crore|lakh|bn|tn)\b"
    r"|"
    r"\b\d+(?:[.,]\d)*\s*(?:billion|million|trillion|crore|lakh|bn|tn)\b"
    r")",
    re.IGNORECASE,
)


def _is_market_size_figure(text: str) -> bool:
    lowered = text.lower()
    if any(term in lowered for term in MARKET_SIZE_EXCLUDED_TERMS):
        return False
    return "market" in lowered and bool(_MARKET_SIZE_AMOUNT.search(text))


_QUANTITATIVE_FIGURE = re.compile(
    r"(?:"
    r"(?:US\$|\$|USD|NPR|INR|Rs\.?|₹|₨|EUR|€|GBP|£|JPY|CNY|¥)"
    r"\s*\d+(?:[.,]\d)*(?:\s*(?:–|-)\s*(?:(?:US\$|\$|USD|NPR|INR|Rs\.?|₹|₨|EUR|€|GBP|£|JPY|CNY|¥)\s*)?\d+(?:[.,]\d)*)?"
    r"|"
    r"\d+(?:[.,]\d)*\s*(?:thousand|million|billion|trillion|lakh|crore|bn|tn)\b"
    r"|"
    r"\d+(?:[.,]\d)*%"
    r")",
    re.IGNORECASE,
)
_WEBSITE_NAME = re.compile(
    r"\b[A-Za-z0-9-]+\.(?:com|io|org|info|net|in|np)\b",
    re.IGNORECASE,
)


def _cited_publisher_names(ids: list[str], research: ResearcherOutput) -> set[str]:
    by_id = {f.finding_id: f for f in research.findings}
    sources = {s.source_id: s for s in research.sources}
    names: set[str] = set()
    for finding_id in ids:
        finding = by_id.get(finding_id)
        if not finding:
            continue
        for source_id in finding.source_ids:
            source = sources.get(source_id)
            if source and source.publisher:
                names.add(source.publisher.strip().lower())
    return names


def _source_names_match(text: str, publisher_names: set[str]) -> bool:
    websites = {name.lower() for name in _WEBSITE_NAME.findall(text)}
    return all(
        any(
            website == publisher.strip().lower()
            or website in publisher.strip().lower()
            or publisher.strip().lower() in website
            for publisher in publisher_names
        )
        for website in websites
    )


def _compute_missing(research) -> list[str]:
    local = [
        f"{f.claim} {f.evidence}".lower()
        for f in research.findings
        if f.scope == "target_geography"
    ]
    missing = []
    for metric in research.plan.required_metrics:
        m = metric.lower()
        metric_name = next(
            (name for name in METRIC_KEYWORDS if name in m),
            None,
        )
        keys = METRIC_KEYWORDS.get(metric_name, ()) if metric_name else ()
        covered = any(
            _QUANTITATIVE_FIGURE.search(text)
            and any(k in text for k in keys)
            and (
                metric_name != "total addressable market"
                or _is_market_size_finding(text)
            )
            for text in local
        )
        if not covered:
            missing.append(metric)
    return missing


def _strip_ids(text: str) -> str:
    return re.sub(r"\s*\(\s*F\d+(?:\s*,\s*F\d+)*\s*\)", "", text).strip()


class _BriefDraft(StrictModel):
    """Step 1 output. No summary and no source_ids: code handles those."""

    subject: str
    competitors: list[BriefCompetitor] = Field(default_factory=list)
    key_observations: list[KeyObservation] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


class _SummaryDraft(StrictModel):
    executive_summary: str


def _findings_block(research: ResearcherOutput) -> str:
    sources = {s.source_id: s for s in research.sources}
    lines = []
    for f in research.findings:
        src = ", ".join(
            f"{sources[sid].publisher or sources[sid].title} ({sid})"
            for sid in f.source_ids
            if sid in sources
        )
        lines.append(
            f"[{f.finding_id}] scope={f.scope} | confidence={f.confidence} | "
            f"type={f.information_type} | source: {src}\n"
            f"CLAIM: {f.claim}\nEVIDENCE: {f.evidence}"
        )
    return "\n\n".join(lines) if lines else "(no findings were collected)"


def _sanitize(draft: _BriefDraft, research: ResearcherOutput):
    by_id = {f.finding_id: f for f in research.findings}
    geo = research.plan.geography

    competitors: list[BriefCompetitor] = []
    seen_names: set[str] = set()
    for c in draft.competitors:
        ids = [i for i in c.evidence_finding_ids if i in by_id]
        support = " ".join(
            f"{by_id[i].claim} {by_id[i].evidence}" for i in ids
        ).lower()
        if c.name.strip().lower() not in support:
            print(f"[debug] dropped competitor '{c.name}': not named in cited findings")
            continue
        if not any(by_id[i].scope == "target_geography" for i in ids):
            print(f"[debug] dropped competitor '{c.name}': no valid in-geography evidence")
            continue
        if c.name.strip().lower() in seen_names:
            continue
        seen_names.add(c.name.strip().lower())
        description = _strip_ids(c.description)
        if _WEBSITE_NAME.findall(description):
            publisher_names = _cited_publisher_names(ids, research)
            if not _source_names_match(description, publisher_names):
                print(
                    f"[debug] dropped competitor '{c.name}': website name "
                    "is not from a cited source"
                )
                continue
        competitors.append(
            c.model_copy(
                update={
                    "description": description,
                    "evidence_finding_ids": ids,
                }
            )
        )
    competitors = competitors[:2]

    observations: list[KeyObservation] = []
    for o in draft.key_observations:
        ids = [i for i in o.evidence_finding_ids if i in by_id]
        if not ids:
            print(f"[debug] dropped observation with no valid citation: {o.text[:70]}")
            continue
        scopes = {by_id[i].scope for i in ids}
        if "target_geography" in scopes and "broader_context" in scopes:
            print(
                f"[debug] dropped observation mixing target-geography and "
                f"broader-context findings: {o.text[:70]}"
            )
            continue
        text = _strip_ids(o.text.strip())
        if _WEBSITE_NAME.findall(text):
            publisher_names = _cited_publisher_names(ids, research)
            if not _source_names_match(text, publisher_names):
                print(
                    f"[debug] dropped observation naming a website not from "
                    f"a cited source: {text[:70]}"
                )
                continue
        if all(by_id[i].scope == "broader_context" for i in ids):
            if geo and geo.lower() in text.lower():
                print(f"[debug] dropped observation attributing broader data to {geo}: {text[:70]}")
                continue
            text = f"Not specific to {geo or 'the target'}: {text}"
        observations.append(
            o.model_copy(update={"text": text, "evidence_finding_ids": ids})
        )
    observations = observations[:MAX_OBSERVATIONS]

    if not observations:
        raise LLMError("Analyst produced no observations backed by valid findings.")

    size_sources = _market_size_sources(research)
    notes = []
    if len(size_sources) >= 2:
        notes.append(_comparability_sentence(size_sources))
    missing = list(
        dict.fromkeys(_compute_missing(research) + notes + research.missing_information)
    )
    if competitors:
        competition_questions = {
            q.question
            for q in research.plan.research_questions
            if q.research_dimension.lower() == "competition"
        }
        missing = [
            gap
            for gap in missing
            if not any(question.lower() in gap.lower() for question in competition_questions)
        ]
    return draft.subject, competitors, observations, missing


def _numbers(text: str) -> set[str]:
    return set(_NUMBER.findall(text))


def _lowercase_initial(text: str) -> str:
    if len(text) >= 2 and text[1].islower():
        return text[0].lower() + text[1:]
    return text


def _capitalise_subject(subject: str, geography: str | None = None) -> str:
    title = subject.strip()
    if geography:
        geography = geography.strip()
        if geography and geography.lower() not in title.lower():
            title = f"{title} in {geography}"
    return title[:1].upper() + title[1:]


def _hedge_low_confidence_text(
    text: str, finding_ids: list[str], research: ResearcherOutput
) -> str:
    low_text = text.lower()
    if any(word.lower() in low_text for word in HEDGE_WORDS):
        return text

    by_id = {f.finding_id: f for f in research.findings}
    sources = {s.source_id: s for s in research.sources}
    for finding_id in finding_ids:
        finding = by_id.get(finding_id)
        if not finding or finding.confidence != "low":
            continue
        for source_id in finding.source_ids:
            source = sources.get(source_id)
            if source:
                publisher = source.publisher or source.title
                return f"According to {publisher}, {_lowercase_initial(text)}"
    return text


UNNAMED_REFERENCES = (
    "the advisory",
    "the report",
    "the study",
    "the article",
    "the paper",
    "the order",
    "the notification",
    "this report",
    "this study",
    "this article",
    "this paper",
)


def _attributed_reference_text(
    text: str,
    finding_ids: list[str],
    research: ResearcherOutput,
) -> str:
    lowered = text.lstrip().lower()
    if not any(lowered.startswith(reference) for reference in UNNAMED_REFERENCES):
        return text

    by_id = {f.finding_id: f for f in research.findings}
    sources = {s.source_id: s for s in research.sources}
    for finding_id in finding_ids:
        finding = by_id.get(finding_id)
        if not finding:
            continue
        for source_id in finding.source_ids:
            source = sources.get(source_id)
            if source:
                publisher = source.publisher or source.title
                return f"According to {publisher}, {_lowercase_initial(text.lstrip())}"
    return text


def _fallback_summary(research, competitors, local_observations) -> str:
    parts = []
    for observation in local_observations[:4]:
        text = observation.text.strip()
        if text:
            parts.append(text if text.endswith(".") else text + ".")

    if competitors:
        names = ", ".join(c.name for c in competitors)
        parts.append(f"The brief lists {names} as competitors.")

    parts.append(
        "Several expected metrics could not be found; see the missing information section."
    )
    size_sources = _market_size_sources(research)
    if len(size_sources) >= 2:
        parts.append(_comparability_sentence(size_sources))
    return " ".join(parts)


def _market_size_sources(research: ResearcherOutput) -> list[str]:
    pubs = {s.source_id: (s.publisher or s.title) for s in research.sources}
    size_sources = {
        pubs[sid]
        for f in research.findings
        if f.scope == "target_geography"
        and _is_market_size_figure(f"{f.claim} {f.evidence}")
        for sid in f.source_ids
        if sid in pubs
    }
    return sorted(size_sources)


def _comparability_sentence(size_sources: list[str]) -> str:
    publishers = ", ".join(size_sources)
    return (
        f"Market-size figures come from several sources ({publishers}) "
        "that may use different definitions and are not directly comparable."
    )


def _ensure_comparability_caveat(summary: str, size_sources: list[str]) -> str:
    if len(size_sources) < 2:
        return summary

    lowered = summary.lower()
    if "sources" in lowered and all(
        publisher.lower() in lowered for publisher in size_sources
    ) and "not directly comparable" in lowered:
        return summary

    sentence = _comparability_sentence(size_sources)
    return f"{summary.rstrip()} {sentence}"


def _summary_covers_brief(
    summary: str,
    competitors: list,
    observations: list,
) -> bool:
    lowered = summary.lower()
    if competitors and not any(
        competitor.name.lower() in lowered for competitor in competitors
    ):
        return False

    regulation_words = ("regulation", "rules", "safety", "permits", "compliance")
    has_regulation = any(
        any(word in observation.text.lower() for word in regulation_words)
        for observation in observations
    )
    if has_regulation and not any(word in lowered for word in regulation_words):
        return False
    return True


def _write_summary(
    query, research, competitors, observations, missing, feedback
) -> str:
    geo = research.plan.geography or research.plan.subject
    label = f"Not specific to {geo}"
    local = [o for o in observations if not o.text.startswith(label)]
    has_context = len(local) < len(observations)

    obs_text = "\n".join(f"- {o.text}" for o in local) or "(none)"
    comp_text = "\n".join(f"- {c.name}: {c.description}" for c in competitors) or "(none)"
    critic = ""
    if feedback:
        critic = (
            "\nThe critic raised these issues about the previous draft; make sure the "
            f"summary does not repeat them:\n{json.dumps(feedback.issues)}\n"
        )
    context_rule = (
        f"- Add one sentence saying the only market figures found are global or "
        f"regional, not specific to {geo}. Do not state those figures.\n"
        if has_context
        else ""
    )

    prompt = f"""Research request: "{query}"

Write an executive summary of 3 to 4 sentences, using ONLY the material below.
Write at most four sentences.

OBSERVATIONS ABOUT {geo.upper()}:
{obs_text}

COMPETITORS:
{comp_text}

GAPS:
{json.dumps(missing)}
{critic}
Rules:
- Restate only what the observations and competitors say. Add no new facts,
  trends or conclusions. Do not call the market "growing" or "large".
- Use only numbers that appear in the observations above. Do not add any other number.
- When two figures come from different publishers, name the publisher next to each figure.
- Name the publisher next to every market figure.
- Attribute a figure only to the publisher shown for the finding it comes from.
- Never present figures from two publishers as one continuous trend or one
  "growth from X to Y" statement.
- Write one sentence per topic area present in the observations: market figures,
  competitors, business model or economics, regulation, and customers.
{context_rule}- Say plainly what is not known about {geo} (use the GAPS).
Return JSON: {{"executive_summary": "..."}}"""

    summary = generate_structured(prompt, _SummaryDraft, system=SYSTEM).executive_summary
    size_sources = _market_size_sources(research)

    if not _numbers(summary) <= _numbers(obs_text + comp_text):
        print("[debug] summary contained numbers not in the observations; using fallback")
        summary = _fallback_summary(research, competitors, local)
    if not _summary_covers_brief(summary, competitors, observations):
        print("[debug] summary omitted brief topics; using fallback")
        summary = _fallback_summary(research, competitors, local)
    summary = _ensure_comparability_caveat(summary, size_sources)

    attribution_brief = ResearchBrief(
        subject=_capitalise_subject(research.plan.subject, research.plan.geography),
        executive_summary=summary,
        competitors=competitors,
        key_observations=observations,
        source_ids=list(
            {sid for f in research.findings for sid in f.source_ids}
        ),
        missing_information=missing,
    )
    if _attribution_issues(attribution_brief, research):
        print("[debug] summary had attribution problems; using fallback")
        summary = _fallback_summary(research, competitors, local)
    return _ensure_comparability_caveat(summary, size_sources)


def run_analyst(
    query: str,
    research: ResearcherOutput,
    previous_draft: ResearchBrief | None = None,
    feedback: CriticFeedback | None = None,
) -> ResearchBrief:
    geo = research.plan.geography or research.plan.subject

    revision = ""
    if previous_draft and feedback:
        revision = f"""
THIS IS A REVISION. Your previous draft was reviewed.
PREVIOUS DRAFT:
{previous_draft.model_dump_json(indent=2)}

CRITIC ISSUES:
{json.dumps(feedback.issues, indent=2)}

REVISION INSTRUCTIONS (follow all of them):
{json.dumps(feedback.revision_instructions, indent=2)}

Fix the problems using only the findings below. Do not add unsupported content.
"""

    prompt = f"""Research request: "{query}"
{revision}
FINDINGS (the only evidence you may use):
{_findings_block(research)}

RESEARCH GAPS reported by the researcher:
{json.dumps(research.missing_information, indent=2)}

Return JSON with these fields:
- subject: the topic of the brief.
- competitors: at most 2. Only companies that the findings say operate in {geo}.
  Each has name, description and evidence_finding_ids. The description must restate
  only what the cited finding says about THAT company (if a finding only mentions
  it in passing, say so), and must hedge when the finding is low confidence or from
  a social post. Never use broader_context findings. Fewer than 2, or none, is fine.
- key_observations: 3 to {MAX_OBSERVATIONS} items. Each has text (one claim) and
  evidence_finding_ids (1 to 3 IDs). Choose findings that tell the reader
  something useful about the market; skip findings that only describe a study's
  method or variables, or that restate each other.
- missing_information: plain-language list of what the evidence does not cover,
  including required metrics with no findings: {", ".join(research.plan.required_metrics)}.

Rules:
- Cite only finding IDs that exist above.
- scope=broader_context findings are about the world or other regions, NOT {geo}.
  Say what they apply to (for example "Globally, ..." or "In Asia Pacific, ...") and
  never write {geo} in an observation based only on them.
- Quote figures exactly as given. Do not calculate, extrapolate or round.
- If two findings give different figures for the same thing, say they differ.
- Do not state a trend, growth or size for {geo} unless a {geo}-scoped finding says it.
- Never write finding IDs (like F1 or F2) inside any text field. To compare two
  figures, cite both IDs in evidence_finding_ids and refer to each figure by its
  publisher name shown in its finding, and never name a publisher that does not
  appear in the findings.
- A competitor description may only contain facts the cited finding states about
  THAT company. Never attach a market-wide figure to a company. If no finding names
  a company, do not list it as a competitor.
- Do not make recommendations or predictions."""

    draft = generate_structured(prompt, _BriefDraft, system=SYSTEM)
    subject, competitors, observations, missing = _sanitize(draft, research)
    if feedback:
        competitors = [
            c.model_copy(
                update={
                    "description": _attributed_reference_text(
                        _hedge_low_confidence_text(
                            c.description,
                            c.evidence_finding_ids,
                            research,
                        ),
                        c.evidence_finding_ids,
                        research,
                    )
                }
            )
            for c in competitors
        ]
        observations = [
            o.model_copy(
                update={
                    "text": _attributed_reference_text(
                        _hedge_low_confidence_text(
                            o.text,
                            o.evidence_finding_ids,
                            research,
                        ),
                        o.evidence_finding_ids,
                        research,
                    )
                }
            )
            for o in observations
        ]
    summary = _write_summary(query, research, competitors, observations, missing, feedback)

    by_id = {f.finding_id: f for f in research.findings}
    cited = {i for c in competitors for i in c.evidence_finding_ids}
    cited |= {i for o in observations for i in o.evidence_finding_ids}
    source_ids = sorted(
        {sid for i in cited for sid in by_id[i].source_ids},
        key=lambda s: int(s[1:]),
    )

    brief = ResearchBrief(
        subject=_capitalise_subject(
            research.plan.subject,
            research.plan.geography,
        ),
        executive_summary=summary,
        competitors=competitors,
        key_observations=observations,
        source_ids=source_ids,
        missing_information=missing,
    )
    problems = check_brief_references(brief, research)
    if problems:
        raise LLMError(f"Brief has unresolved references: {problems}")
    return brief