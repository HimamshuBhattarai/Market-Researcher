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
from market_research.services.llm import LLMError, generate_structured

SYSTEM = (
    "You are a market research analyst writing a short, evidence-backed brief. "
    "You only use the findings provided. You never use outside knowledge and "
    "never invent facts, figures, companies or sources."
)

MAX_OBSERVATIONS = 6
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


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
        if not any(by_id[i].scope == "target_geography" for i in ids):
            print(f"[debug] dropped competitor '{c.name}': no valid in-geography evidence")
            continue
        if c.name.strip().lower() in seen_names:
            continue
        seen_names.add(c.name.strip().lower())
        competitors.append(c.model_copy(update={"evidence_finding_ids": ids}))
    competitors = competitors[:2]

    observations: list[KeyObservation] = []
    for o in draft.key_observations:
        ids = [i for i in o.evidence_finding_ids if i in by_id]
        if not ids:
            print(f"[debug] dropped observation with no valid citation: {o.text[:70]}")
            continue
        text = o.text.strip()
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

    missing = list(
        dict.fromkeys(draft.missing_information + research.missing_information)
    )
    return draft.subject, competitors, observations, missing


def _numbers(text: str) -> set[str]:
    return set(_NUMBER.findall(text))


def _fallback_summary(research, competitors) -> str:
    geo = research.plan.geography or research.plan.subject
    parts = []
    if competitors:
        names = ", ".join(c.name for c in competitors)
        parts.append(f"Sources mention {names} as food delivery operators in {geo}.")
    parts.append(
        "This brief is limited to the cited findings below; observations marked "
        f"'Not specific to {geo}' are broader context only. See the missing "
        "information section for what the evidence does not cover."
    )
    return " ".join(parts)


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
- Do not include any numbers or years. Figures belong in the observations.
{context_rule}- Say plainly what is not known about {geo} (use the GAPS).
Return JSON: {{"executive_summary": "..."}}"""

    summary = generate_structured(prompt, _SummaryDraft, system=SYSTEM).executive_summary

    if re.search(r"\d", summary):
        print("[debug] summary contained digits; using fallback")
        return _fallback_summary(research, competitors)
    return summary


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
- Do not make recommendations or predictions."""

    draft = generate_structured(prompt, _BriefDraft, system=SYSTEM)
    subject, competitors, observations, missing = _sanitize(draft, research)
    summary = _write_summary(query, research, competitors, observations, missing, feedback)

    by_id = {f.finding_id: f for f in research.findings}
    cited = {i for c in competitors for i in c.evidence_finding_ids}
    cited |= {i for o in observations for i in o.evidence_finding_ids}
    source_ids = sorted(
        {sid for i in cited for sid in by_id[i].source_ids},
        key=lambda s: int(s[1:]),
    )

    brief = ResearchBrief(
        subject=research.plan.subject,
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