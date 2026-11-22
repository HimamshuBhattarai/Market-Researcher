import argparse
import json
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from market_research.agents.orchestrator import run_pipeline


DEFAULT_QUERY = "Analyze the food delivery market in Nepal."


def _cited_findings(state, brief):
    """Return findings cited by the final brief."""
    if not state.research_output or not brief:
        return []

    cited_ids = {
        finding_id
        for competitor in brief.competitors
        for finding_id in competitor.evidence_finding_ids
    }
    cited_ids.update(
        finding_id
        for observation in brief.key_observations
        for finding_id in observation.evidence_finding_ids
    )
    by_id = {
        finding.finding_id: finding
        for finding in state.research_output.findings
    }
    return [
        {
            "claim": by_id[finding_id].claim,
            "evidence": by_id[finding_id].evidence,
            "scope": by_id[finding_id].scope,
            "confidence": by_id[finding_id].confidence,
            "source_ids": by_id[finding_id].source_ids,
        }
        for finding_id in sorted(cited_ids)
        if finding_id in by_id
    ]


def _brief_sources(state, brief):
    """Return sources listed by the final brief."""
    if not state.research_output or not brief:
        return []

    listed = set(brief.source_ids)
    return [
        source.model_dump(mode="json")
        for source in state.research_output.sources
        if source.source_id in listed
    ]


def _write_json_output(
    state,
    brief,
    output_dir: Path,
    query: str,
    research_mode: str,
) -> Path:
    payload = {
        "query": query,
        "research_mode": research_mode,
        "status": state.status,
        "revision_count": state.revision_count,
        "critic_feedback": (
            state.critic_feedback.model_dump(mode="json")
            if state.critic_feedback
            else None
        ),
        "final_brief": brief.model_dump(mode="json") if brief else None,
        "findings": _cited_findings(state, brief),
        "sources": _brief_sources(state, brief),
    }
    path = output_dir / "sample_output.json"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _source_link(source) -> str:
    publisher = source.publisher or source.title
    return f"[{publisher}]({source.url}) (accessed {source.accessed_date})"


def _write_markdown_output(
    state,
    brief,
    output_dir: Path,
    query: str,
    research_mode: str,
) -> Path:
    if brief:
        lines = [
            f"# {brief.subject}",
            "",
            f"*{query}*",
            "",
        ]
        if research_mode == "resumed":
            lines.extend(["*MODE: RESUMED from saved research*", ""])
        lines.extend([
            "## Executive summary",
            "",
            brief.executive_summary,
            "",
            "## Competitors",
            "",
        ])
        if brief.competitors:
            for competitor in brief.competitors:
                lines.append(
                    f"- **{competitor.name}:** {competitor.description}"
                )
        else:
            lines.append("None supported by the evidence.")

        lines.extend(["", "## Key observations", ""])
        for observation in brief.key_observations:
            lines.append(f"- {observation.text}")

        lines.extend(["", "## Sources", ""])
        listed = set(brief.source_ids)
        sources = (
            [
                source
                for source in state.research_output.sources
                if source.source_id in listed
            ]
            if state.research_output
            else []
        )
        if sources:
            for source in sources:
                lines.append(f"- {_source_link(source)}")
        else:
            lines.append("- None.")

        lines.extend(["", "## Missing information and limitations", ""])
        if brief.missing_information:
            for item in brief.missing_information:
                lines.append(f"- {item}")
        else:
            lines.append("- None.")
        markdown = "\n".join(lines) + "\n"
    else:
        lines = [
            f"# {query}",
            "",
            f"*{query}*",
            "",
        ]
        if research_mode == "resumed":
            lines.extend(["*MODE: RESUMED from saved research*", ""])
        lines.extend([
            "## Executive summary",
            "",
            "No brief was produced.",
            "",
            "## Competitors",
            "",
            "None supported by the evidence.",
            "",
            "## Key observations",
            "",
            "None.",
            "",
            "## Sources",
            "",
            "None.",
            "",
            "## Missing information and limitations",
            "",
            "- No final brief was produced.",
        ])
        markdown = "\n".join(lines)

    path = output_dir / "sample_output.md"
    path.write_text(markdown, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="*", default=[DEFAULT_QUERY])
    parser.add_argument("--research", type=Path, default=None)
    arguments = parser.parse_args(sys.argv[1:] if argv is None else argv)
    query = " ".join(arguments.query).strip() or DEFAULT_QUERY
    research_path = arguments.research

    state = run_pipeline(
        query,
        on_progress=print,
        research_path=research_path,
    )
    research_mode = "resumed" if research_path else "live"
    print(f"\nStatus: {state.status}")
    print(f"Revision count: {state.revision_count}")
    print(f"Error: {state.error or 'None'}")

    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = _write_json_output(
        state,
        state.final_brief,
        output_dir,
        query,
        research_mode,
    )
    markdown_path = _write_markdown_output(
        state,
        state.final_brief,
        output_dir,
        query,
        research_mode,
    )

    print(f"JSON output: {json_path}")
    print(f"Markdown output: {markdown_path}")
    return 1 if state.status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
