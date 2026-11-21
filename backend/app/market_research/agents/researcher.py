import re
from datetime import date
from typing import Callable, Literal
from urllib.parse import urlparse

from pydantic import Field

from market_research.config import MAX_FINDINGS_PER_QUESTION, RESULTS_PER_QUERY
from market_research.schemas import (
    ExtractedFinding,
    ResearchFinding,
    ResearchPlan,
    ResearchQuestion,
    ResearcherOutput,
    Source,
)
from market_research.schemas.base import StrictModel
from market_research.research.planner import make_plan
from market_research.research.search import SearchError, search_web
from market_research.services.llm import LLMError, generate_structured

SYSTEM = (
    "You are a careful research assistant. You only report what the provided "
    "sources explicitly state. You never use outside knowledge."
)

LOW_TRUST_DOMAINS = (
    "facebook.com", "linkedin.com", "scribd.com", "medium.com",
    "quora.com", "reddit.com", "youtube.com", "x.com", "twitter.com",
)
GEO_MIN_MENTIONS = 2
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


class _SearchQuery(StrictModel):
    query: str


class _Extraction(StrictModel):
    findings: list[ExtractedFinding] = Field(default_factory=list)


class _Verdict(StrictModel):
    index: int
    verdict: Literal["supported", "unsupported", "off_topic"]


class _Verification(StrictModel):
    verdicts: list[_Verdict] = Field(default_factory=list)


class _SourceRegistry:
    """Code assigns source IDs, dates and publishers. The LLM never does."""

    def __init__(self) -> None:
        self.sources: dict[str, Source] = {}
        self.texts: dict[str, str] = {}
        self._by_url: dict[str, str] = {}

    def add(self, result) -> str:
        if result.url in self._by_url:
            return self._by_url[result.url]
        sid = f"S{len(self.sources) + 1}"
        host = urlparse(result.url).netloc.removeprefix("www.") or None
        self.sources[sid] = Source(
            source_id=sid,
            title=result.title,
            url=result.url,
            publisher=host,
            publication_date=result.published,
            accessed_date=date.today().isoformat(),
            source_type="web page",
        )
        self.texts[sid] = result.text
        self._by_url[result.url] = sid
        return sid


# ---------- deterministic checks ----------

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _evidence_supported(evidence: str, source_ids: list[str], texts: dict) -> bool:
    """The quoted evidence must actually appear in a cited source."""
    parts = [
        _norm(p).strip(" \"'“”‘’.")
        for p in re.split(r"\.\.\.|…", evidence)
    ]
    parts = [p for p in parts if p]
    if not parts:
        return False
    return any(
        all(p in _norm(texts[sid]) for p in parts) for sid in source_ids if sid in texts
    )


def _numbers_grounded(claim: str, evidence: str) -> bool:
    """Every number in the claim must appear in the evidence passage."""
    return all(n in evidence for n in _NUMBER.findall(claim))


def _geo_specific(
    geography: str, evidence: str, cited: list[str], registry: _SourceRegistry
) -> bool:
    g = geography.lower()
    if g in evidence.lower():
        return True
    for sid in cited:
        src = registry.sources[sid]
        if g in src.title.lower() or g in src.url.lower():
            return True
        if registry.texts[sid].lower().count(g) >= GEO_MIN_MENTIONS:
            return True
    return False


def _low_trust(cited: list[str], registry: _SourceRegistry) -> bool:
    for sid in cited:
        host = (registry.sources[sid].publisher or "").lower()
        if any(host == d or host.endswith("." + d) for d in LOW_TRUST_DOMAINS):
            return True
    return False


# ---------- LLM steps ----------

def _build_search_query(
    plan: ResearchPlan, q: ResearchQuestion, retry: bool = False
) -> str:
    extra = (
        "A previous search returned mostly global pages. Write a DIFFERENT query "
        "that names local companies, cities or local publishers and academic "
        "studies, not global market-research report titles.\n"
        if retry
        else "Prefer wording that local news, company and academic pages would use "
        "over the titles of global market-research reports.\n"
    )
    prompt = f"""Write ONE concise web search query (3 to 10 keywords) that is likely
to find specific, factual, citable pages for this research question.

Subject: {plan.subject}
Geography: {plan.geography or "not specified"}
Question: {q.question}
Metrics of interest: {", ".join(q.relevant_metrics) or "none"}

The query MUST contain the geography name.
{extra}Do not include any answer in the query."""
    return generate_structured(prompt, _SearchQuery, system=SYSTEM).query


def _verify_findings(
    plan: ResearchPlan, q: ResearchQuestion, candidates: list[ExtractedFinding]
) -> list[ExtractedFinding]:
    """LLM judge: keep only findings whose evidence supports the claim AND
    which actually help answer the question."""
    listing = "\n\n".join(
        f"[{i}] CLAIM: {f.claim}\n    EVIDENCE: {f.evidence}"
        for i, f in enumerate(candidates)
    )
    prompt = f"""Research question: {q.question}
Subject: {plan.subject}

Judge each finding below. For each index give one verdict:
- "supported": the EVIDENCE passage directly states what the CLAIM says, with no
  added facts, entities, numbers or change of topic (for example, evidence about
  ride-hailing or transport does not support a claim about food delivery), AND the
  claim helps answer the research question.
- "unsupported": the claim says more than, or something different from, the evidence.
- "off_topic": the claim is supported but does not answer THIS research question.
  A market-size or revenue figure does not answer a question about competitors,
  business models, customers, growth rate or economics. A competitor question needs
  a finding that names specific companies. A growth question needs a growth rate or
  year-over-year change. When unsure, choose off_topic. This also includes
  descriptions of a study's method or variables, a company's own marketing, generic
  definitions, and figures for another region that are not useful context.

Return exactly: {{"verdicts": [{{"index": 0, "verdict": "supported"}}, ...]}}
Include every index.

FINDINGS:
{listing}"""
    result = generate_structured(prompt, _Verification, system=SYSTEM)
    verdicts = {v.index: v.verdict for v in result.verdicts}
    return [f for i, f in enumerate(candidates) if verdicts.get(i) == "supported"]


def _extract_findings(
    plan: ResearchPlan,
    q: ResearchQuestion,
    source_ids: list[str],
    registry: _SourceRegistry,
) -> tuple[list[ExtractedFinding], int]:
    geo = plan.geography or plan.subject
    blocks = "\n\n".join(
        f"[{sid}] {registry.sources[sid].title} ({registry.sources[sid].url})\n"
        f"{registry.texts[sid]}"
        for sid in source_ids
    )
    prompt = f"""Research question: {q.question}
Subject: {plan.subject}

SOURCES:
{blocks}

Extract up to {MAX_FINDINGS_PER_QUESTION} findings that help answer the question.

Rules:
- Return exactly this shape: {{"findings": [ ...finding objects... ]}}
- Use ONLY information explicitly stated in the sources above.
- evidence: copy a short passage VERBATIM from the cited source. Do not paraphrase.
- claim: one sentence that restates ONLY what the evidence passage says. Do not add
  entities, numbers or topics that are not in the passage. If the passage is about
  a different topic (for example transport or ride-hailing), do not turn it into a
  claim about {plan.domain}.
- source_ids: IDs from the sources above (e.g. "S1") that contain the evidence.
- scope: "target_geography" ONLY if the evidence passage or the source title is
  specifically about {geo}. Use "broader_context" for the whole world, other
  countries or other regions.
- A broader_context claim must name what it applies to (e.g. "Globally, ...").
- Prefer findings about {geo}. Include at most one broader_context finding, and
  only if it is genuinely useful context for the question.
- information_type: "fact" if stated as established fact, "estimate" if a
  forecast or approximate figure, "interpretation" if opinion or analysis.
- confidence: "high" only for specific claims from official or reputable sources;
  at most "medium" for blogs, social posts or marketing pages; "low" for vague or
  promotional claims.
- Never include a number that is not in the evidence.
- Only extract findings that answer THIS question. Do not return a market-size
  figure for a question about competitors, business models, customers, growth or
  economics. Returning an empty list is better than returning an off-topic finding.
- If the sources contain nothing relevant, return {{"findings": []}}."""

    extraction = generate_structured(prompt, _Extraction, system=SYSTEM)

    candidates: list[ExtractedFinding] = []
    dropped = 0
    allowed = set(source_ids)

    for f in extraction.findings[:MAX_FINDINGS_PER_QUESTION]:
        cited = [s for s in f.source_ids if s in allowed]
        if (
            not cited
            or not _evidence_supported(f.evidence, cited, registry.texts)
            or not _numbers_grounded(f.claim, f.evidence)
        ):
            dropped += 1
            continue

        updates: dict = {"source_ids": cited}
        if (
            f.scope == "target_geography"
            and plan.geography
            and not _geo_specific(plan.geography, f.evidence, cited, registry)
        ):
            updates["scope"] = "broader_context"
        if _low_trust(cited, registry):
            updates["confidence"] = "low"
        candidates.append(f.model_copy(update=updates))

    if not candidates:
        return [], dropped

    verified = _verify_findings(plan, q, candidates)
    dropped += len(candidates) - len(verified)
    return verified, dropped


# ---------- orchestration of the research stage ----------

def run_researcher(
    query: str, on_progress: Callable[[str], None] | None = None
) -> ResearcherOutput:
    progress = on_progress or (lambda msg: None)

    progress("Planning research...")
    plan = make_plan(query)
    geo_label = plan.geography or "subject"

    registry = _SourceRegistry()
    kept: list[tuple[str, ExtractedFinding]] = []  # (question_id, finding)
    missing: list[str] = []
    total_dropped = 0
    total = len(plan.research_questions)
    seen_evidence: set[str] = set()

    for i, q in enumerate(plan.research_questions, start=1):
        progress(f"Researching {i}/{total}: {q.research_dimension}")
        found_q: list[ExtractedFinding] = []
        errored = False

        for attempt in (1, 2):
            if attempt == 2:
                if any(f.scope == "target_geography" for f in found_q):
                    break
                progress(f"  Retrying with a different query: {q.research_dimension}")
            try:
                search_query = _build_search_query(plan, q, retry=(attempt == 2))
                results = search_web(search_query, RESULTS_PER_QUERY)
                if not results:
                    continue
                source_ids = [registry.add(r) for r in results]
                found, dropped = _extract_findings(plan, q, source_ids, registry)
                total_dropped += dropped
                found_q.extend(found)
            except (SearchError, LLMError) as e:
                print(f"[debug] {q.question_id} attempt {attempt} failed: {e}")
                errored = True
                break

        # de-duplicate, prefer geography-specific findings, cap per question
        found_q.sort(key=lambda f: f.scope != "target_geography")
        unique: list[ExtractedFinding] = []
        for f in found_q:
            evidence_key = _norm(f.evidence)
            if evidence_key not in seen_evidence:
                seen_evidence.add(evidence_key)
                unique.append(f)
            if len(unique) >= MAX_FINDINGS_PER_QUESTION:
                break
        found_q = unique

        if not found_q:
            if errored:
                missing.append(f"Research step failed for: {q.question}")
            else:
                missing.append(f"No sourced evidence found for: {q.question}")
        elif not any(f.scope == "target_geography" for f in found_q):
            missing.append(
                f"No {geo_label}-specific evidence found for: {q.question} "
                f"(only broader-context data)"
            )
        kept.extend((q.question_id, f) for f in found_q)

    if total_dropped:
        progress(f"Dropped {total_dropped} finding(s) that failed verification.")

    findings = [
        ResearchFinding(**f.model_dump(), finding_id=f"F{i}", research_question_id=qid)
        for i, (qid, f) in enumerate(kept, start=1)
    ]
    used = {sid for f in findings for sid in f.source_ids}
    sources = sorted(
        (registry.sources[sid] for sid in used), key=lambda s: int(s.source_id[1:])
    )

    progress("Research complete.")
    return ResearcherOutput(
        plan=plan,
        findings=findings,
        sources=sources,
        missing_information=missing,
    )