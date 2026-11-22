import re

from pydantic import Field

from market_research.schemas import CriticFeedback, ResearchBrief, ResearcherOutput
from market_research.schemas.base import StrictModel
from market_research.services.llm import generate_structured

SYSTEM = (
    "You are a strict but fair research reviewer. You check a brief against the "
    "evidence it cites. You do not do new research and never add facts."
)

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_BARE_ID = re.compile(r"\bF\d+\b")
HEDGE_WORDS = (
    "report", "according", "claim", "state", "said", "post",
    "announce", "describ", "suggest", "appears", "may",
)
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
WEBSITE_NAME = re.compile(
    r"\b[A-Za-z0-9-]+\.(?:com|io|org|info|net|in|np)\b",
    re.IGNORECASE,
)
ATTRIBUTION = re.compile(
    r"\bAccording\s+to\s+([A-Za-z0-9.-]+(?:\.[A-Za-z0-9.-]+)*)",
    re.IGNORECASE,
)


def _cited_publishers(ids: list[str], research: ResearcherOutput) -> set[str]:
    by_id = {f.finding_id: f for f in research.findings}
    sources = {s.source_id: s for s in research.sources}
    publishers: set[str] = set()
    for finding_id in ids:
        finding = by_id.get(finding_id)
        if not finding:
            continue
        for source_id in finding.source_ids:
            source = sources.get(source_id)
            if source and source.publisher:
                publishers.add(source.publisher.strip().lower())
    return publishers


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


class _Issue(StrictModel):
    quote: str  # exact text copied from ONE statement in the brief
    unsupported_phrase: str  # exact words from that quote the evidence lacks


class _CriticDraft(StrictModel):
    issues: list[_Issue] = Field(default_factory=list)


def _cited_text(ids: list[str], by_id: dict) -> str:
    return " ".join(f"{by_id[i].claim} {by_id[i].evidence}" for i in ids if i in by_id)


def _statements(brief: ResearchBrief, include_summary: bool = False):
    items = [
        (f"Observation {n}", o.text, o.evidence_finding_ids)
        for n, o in enumerate(brief.key_observations, start=1)
    ]
    items += [
        (f"Competitor '{c.name}'", c.description, c.evidence_finding_ids)
        for c in brief.competitors
    ]
    if include_summary:
        all_ids = sorted({i for _, _, ids in items for i in ids})
        items.append(("Summary", brief.executive_summary, all_ids))
    return items


# ---------- deterministic checks (the dependable part) ----------

def _publisher_findings(research: ResearcherOutput) -> dict[str, list]:
    sources = {s.source_id: s for s in research.sources}
    by_publisher: dict[str, list] = {}
    for finding in research.findings:
        for source_id in finding.source_ids:
            source = sources.get(source_id)
            publisher = (source.publisher or source.title).strip().lower()
            if publisher:
                by_publisher.setdefault(publisher, []).append(finding)
    return by_publisher


def _attribution_issues(
    brief: ResearchBrief,
    research: ResearcherOutput,
) -> list[str]:
    by_publisher = _publisher_findings(research)
    issues: list[str] = []

    statements = _statements(brief, include_summary=True)
    for label, text, own_ids in statements:
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            attribution = ATTRIBUTION.search(sentence)
            if attribution:
                publisher = attribution.group(1).strip().lower().rstrip(".")
                publisher_findings = by_publisher.get(publisher, [])
                support = " ".join(
                    f"{finding.claim} {finding.evidence}"
                    for finding in publisher_findings
                ).lower()

                numbers = _NUMBER.findall(sentence)
                if numbers:
                    missing = [
                        number for number in numbers
                        if number not in support
                    ]
                    if missing:
                        issues.append(
                            f"{label} attributes figures to {publisher}, but "
                            f"{missing} are not supported by that publisher."
                        )
                else:
                    words = [
                        word.lower()
                        for word in re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)
                        if len(word) > 4
                    ]
                    if words:
                        matched = sum(
                            1 for word in words
                            if word in support
                        )
                        if matched * 2 < len(words):
                            issues.append(
                                f"{label} attributes a claim to {publisher}, "
                                "but it is not supported by that publisher."
                            )

                if label != "Summary" and own_ids:
                    own_publishers = set()
                    finding_by_id = {
                        finding.finding_id: finding
                        for finding in research.findings
                    }
                    sources = {source.source_id: source for source in research.sources}
                    for finding_id in own_ids:
                        finding = finding_by_id.get(finding_id)
                        if not finding:
                            continue
                        for source_id in finding.source_ids:
                            source = sources.get(source_id)
                            if source:
                                own_publishers.add(
                                    (source.publisher or source.title).strip().lower()
                                )
                    if publisher not in own_publishers:
                        issues.append(
                            f"{label} starts with According to {publisher}, "
                            "but none of its own cited findings come from that publisher."
                        )

            sentence_numbers = _NUMBER.findall(sentence)
            if label == "Summary" and len(sentence_numbers) >= 1:
                number_publishers: set[str] = set()
                sources = {source.source_id: source for source in research.sources}
                for number in sentence_numbers:
                    supporting_publishers = set()
                    for finding in research.findings:
                        if number not in f"{finding.claim} {finding.evidence}":
                            continue
                        for source_id in finding.source_ids:
                            source = sources.get(source_id)
                            if source:
                                supporting_publishers.add(
                                    (source.publisher or source.title).strip().lower()
                                )
                    number_publishers.update(supporting_publishers)
                if len(number_publishers) > 1:
                    issues.append(
                        f"{label} sentence combines figures from multiple publishers "
                        f"({', '.join(sorted(number_publishers))})."
                    )
    return issues

def _code_checks(brief: ResearchBrief, research: ResearcherOutput) -> list[str]:
    by_id = {f.finding_id: f for f in research.findings}
    issues = []

    # 1. every figure must appear in the cited findings
    for label, text, ids in _statements(brief):
        support = _cited_text(ids, by_id)
        bad = [n for n in _NUMBER.findall(text) if n not in support]
        if bad:
            issues.append(
                f"{label} contains figures not found in its cited findings: {bad}."
            )

    # 2. internal finding IDs must not leak into reader-facing text
    texts = [("Summary", brief.executive_summary)] + [
        (label, text) for label, text, _ in _statements(brief)
    ]
    for label, text in texts:
        if _BARE_ID.search(text):
            issues.append(
                f"{label} contains internal finding IDs (e.g. F1); remove them and "
                f"refer to sources by name instead."
            )
    return issues


def _scope_and_hedge_checks(
    brief: ResearchBrief, research: ResearcherOutput
) -> list[str]:
    by_id = {f.finding_id: f for f in research.findings}
    geo = research.plan.geography or research.plan.subject
    issues = []
    for label, text, ids in _statements(brief):
        cited = [by_id[i] for i in ids if i in by_id]
        if not cited:
            continue
        low = text.lower()
        if any(f.scope == "broader_context" for f in cited) and not low.startswith(
            f"not specific to {geo.lower()}"
        ):
            issues.append(
                f"{label} cites broader-context findings but is not "
                f"labelled 'Not specific to {geo}:'."
            )
        if any(f.confidence == "low" for f in cited) and not any(
            w in low for w in HEDGE_WORDS
        ):
            issues.append(
                f"{label} relies on a low-confidence source but is worded as "
                f"established fact. Attribute it to the source (for example "
                f"'a social media post reports ...')."
            )
        if any(low.lstrip().startswith(reference) for reference in UNNAMED_REFERENCES):
            issues.append(
                f"{label} begins with an unnamed reference; start with "
                f"'According to <publisher>, ' using the first cited source."
            )
        websites = {name.lower() for name in WEBSITE_NAME.findall(text)}
        publishers = _cited_publishers(ids, research)
        unverified = [
            website
            for website in websites
            if not any(
                website == publisher
                or website in publisher
                or publisher in website
                for publisher in publishers
            )
        ]
        if unverified:
            issues.append(
                f"{label} names website(s) not present in its cited source publishers: "
                f"{sorted(unverified)}."
            )
    return issues


# ---------- LLM review (backstop only) ----------

def _citation_block(ids: list[str], research: ResearcherOutput) -> str:
    by_id = {f.finding_id: f for f in research.findings}
    srcs = {s.source_id: s for s in research.sources}
    lines = []
    for i in ids:
        f = by_id.get(i)
        if not f:
            continue
        names = ", ".join(
            srcs[s].publisher or srcs[s].title for s in f.source_ids if s in srcs
        )
        lines.append(
            f"    [{f.finding_id}] scope={f.scope}, confidence={f.confidence}, "
            f"source={names}\n      CLAIM: {f.claim}\n      EVIDENCE: {f.evidence}"
        )
    return "\n".join(lines)


def _brief_block(brief: ResearchBrief, research: ResearcherOutput) -> str:
    parts = [f"SUMMARY:\n{brief.executive_summary}", "COMPETITORS:"]
    for c in brief.competitors:
        parts.append(
            f"- {c.name}: {c.description}\n"
            f"{_citation_block(c.evidence_finding_ids, research)}"
        )
    if not brief.competitors:
        parts.append("(none)")
    parts.append("OBSERVATIONS:")
    for o in brief.key_observations:
        parts.append(f"- {o.text}\n{_citation_block(o.evidence_finding_ids, research)}")
    return "\n".join(parts)


def _phrase_supported(phrase: str, support: str) -> bool:
    words = [
        w for w in re.findall(r"[a-z0-9.,%$]+", phrase) if len(w) > 3 or w.isdigit()
    ]
    return bool(words) and all(w in support for w in words)


def run_critic(
    query: str, research: ResearcherOutput, brief: ResearchBrief
) -> CriticFeedback:
    by_id = {f.finding_id: f for f in research.findings}
    code_issues = (
        _code_checks(brief, research)
        + _scope_and_hedge_checks(brief, research)
        + _attribution_issues(brief, research)
    )

    prompt = f"""Original request: "{query}"

Check this brief against the evidence cited under each statement.

{_brief_block(brief, research)}

Find statements that say something the cited evidence does NOT say (an added fact,
number, cause, company, or a different subject). For each one provide:
- quote: the EXACT wording copied from ONE statement in the brief.
- unsupported_phrase: the exact words inside that quote that the cited evidence
  does not contain.

Rules:
- A statement is fine if its words are supported by the CLAIM or EVIDENCE text
  shown under it, even when worded differently.
- Do not report empty or short sections, missing information, style, or things you
  think the brief could add.
- Do not report scope or confidence problems. Those are checked elsewhere.
- If you find no unsupported statements, return {{"issues": []}}.
Return JSON: {{"issues": [{{"quote": "...", "unsupported_phrase": "..."}}]}}"""

    draft = generate_structured(prompt, _CriticDraft, system=SYSTEM)

    statements = _statements(brief, include_summary=True)
    kept = []
    for i in draft.issues:
        quote = _norm(i.quote).strip(" \"'")
        phrase = _norm(i.unsupported_phrase).strip(" \"'")
        match = next(
            (
                (label, ids)
                for label, text, ids in statements
                if quote and quote in _norm(text)
            ),
            None,
        )
        if len(phrase) < 4 or phrase not in quote or not match:
            print(f"[debug] discarded critic issue (quote not found): {i.quote[:70]}")
            continue
        label, ids = match
        if _phrase_supported(phrase, _norm(_cited_text(ids, by_id))):
            print(f"[debug] discarded critic issue (phrase is in the evidence): {phrase[:70]}")
            continue
        kept.append((label, i.unsupported_phrase))

    advisory = " | ".join(
        f'{l}: "{p}" is not supported by its cited evidence.' for l, p in kept
    )
    instructions = [f"Fix: {c}" for c in code_issues]
    issues = code_issues
    approved = not issues
    assessment = (
        "No material problems found."
        if approved
        else f"{len(issues)} issue(s) found; see issues and revision instructions."
    )
    if advisory:
        assessment += f" Advisory notes (not acted on): {advisory}"

    return CriticFeedback(
        approved=approved,
        overall_assessment=assessment,
        issues=issues,
        revision_instructions=[] if approved else instructions,
    )