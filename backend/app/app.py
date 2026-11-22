import streamlit as st
import os
import requests
from html import escape

from market_research.schemas import WorkflowState
from market_research.agents.orchestrator import run_pipeline
from market_research.config import ConfigError


API_URL = os.getenv("MARKET_RESEARCH_API_URL", "").strip() or None


st.set_page_config(
    page_title="Market Research Brief",
    page_icon=None,
    layout="centered",
)

st.markdown(
    """
    <style>
    :root {
        --page: #FBFAF7;
        --ink: #1E1E1C;
        --muted: #6B6B66;
        --rule: #E4E0D8;
        --navy: #1F3A5F;
        --off-white: #F3F0EA;
        --brick: #8A5A50;
    }

    [data-testid="stAppViewContainer"] {
        background: var(--page);
        color: var(--ink);
    }

    .block-container {
        max-width: 760px;
        margin: 0 auto;
        padding-top: 3.25rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        font-family: "Source Serif 4", Newsreader, Georgia, serif;
        color: var(--ink);
        font-weight: 600;
    }

    html, body, p, label, input, button,
    [data-testid="stMarkdownContainer"] {
        font-family: "Source Sans 3", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    p, [data-testid="stMarkdownContainer"] p {
        font-size: 17px;
        line-height: 1.65;
    }

    #MainMenu, footer,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
        display: none;
    }

    .app-title {
        margin: 0 0 1.25rem 0;
        padding-bottom: 0.65rem;
        border-bottom: 1px solid var(--navy);
        font-family: "Source Serif 4", Newsreader, Georgia, serif;
        font-size: 42px;
        line-height: 1.15;
        font-weight: 600;
    }

    .app-subtitle {
        max-width: 700px;
        margin: 0 0 2.25rem 0;
        color: var(--muted);
        font-size: 15px;
        line-height: 1.6;
    }

    [data-testid="stTextInput"] label,
    [data-testid="stTextInput"] input {
        font-size: 17px;
    }

    [data-testid="stTextInput"] label {
        color: var(--ink);
        font-weight: 600;
    }

    [data-testid="stTextInput"] input {
        min-height: 46px;
        border: 1px solid var(--rule);
        border-radius: 3px;
        background: var(--page);
        color: var(--ink);
    }

    [data-testid="stTextInput"] input:focus {
        border-color: var(--navy);
        box-shadow: 0 0 0 1px var(--navy);
    }

    .stButton > button {
        min-height: 44px;
        border: 1px solid var(--navy);
        border-radius: 3px;
        background: var(--navy);
        color: var(--page);
        box-shadow: none;
        font-size: 16px;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: var(--navy);
        background: var(--navy);
        color: var(--page);
        box-shadow: none;
        filter: brightness(0.88);
    }

    .stDownloadButton > button {
        min-height: 40px;
        border: 1px solid var(--rule);
        border-radius: 3px;
        background: var(--page);
        color: var(--navy);
        box-shadow: none;
        font-size: 14px;
    }

    .stDownloadButton > button:hover {
        border-color: var(--navy);
        background: var(--page);
        color: var(--navy);
        box-shadow: none;
    }

    .run-note {
        margin: 0.65rem 0 0 0;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.5;
    }

    .progress-line {
        display: flex;
        gap: 1.25rem;
        margin: 1.5rem 0 0.35rem 0;
        padding-bottom: 0.65rem;
        border-bottom: 1px solid var(--rule);
        font-size: 15px;
    }

    .progress-stage {
        color: var(--muted);
    }

    .progress-stage.finished {
        color: var(--ink);
        text-decoration: underline;
        text-decoration-thickness: 1px;
        text-underline-offset: 5px;
    }

    .progress-stage.current {
        color: var(--navy);
        font-weight: 700;
    }

    .progress-message {
        min-height: 1.5em;
        margin: 0;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.5;
    }

    .input-error,
    .config-error,
    .failure-notice {
        margin-top: 0.75rem;
        color: var(--brick);
        font-size: 14px;
        line-height: 1.5;
    }

    .failure-notice,
    .config-error {
        padding: 0.75rem 0.9rem;
        border: 1px solid var(--rule);
        border-radius: 3px;
        background: var(--off-white);
    }

    .report-section {
        margin-top: 2.25rem;
        padding-top: 0.9rem;
        border-top: 1px solid var(--rule);
    }

    .report-section h2,
    .report-section h3 {
        margin: 0 0 0.75rem 0;
        font-family: "Source Serif 4", Newsreader, Georgia, serif;
        font-size: 24px;
        line-height: 1.25;
        font-weight: 600;
    }

    .query-line,
    .muted-line,
    .source-line,
    .evidence-label,
    .final-note {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.55;
    }

    .query-line {
        margin: 1.5rem 0 0 0;
        border-top: 1px solid var(--rule);
        padding-top: 0.9rem;
    }

    .mode-notice,
    .failure-notice {
        margin-top: 0.75rem;
        padding: 0.75rem 0.9rem;
        border: 1px solid var(--rule);
        border-radius: 3px;
        background: var(--off-white);
        color: var(--muted);
        font-size: 14px;
        line-height: 1.5;
    }

    .brief-title {
        margin: 1.75rem 0 0.35rem 0;
        font-family: "Source Serif 4", Newsreader, Georgia, serif;
        font-size: 32px;
        line-height: 1.22;
        font-weight: 600;
    }

    .competitor-name {
        font-weight: 700;
    }

    .observation-list,
    .limitation-list,
    .source-list {
        margin: 0;
        padding-left: 1.25rem;
    }

    .observation-list li,
    .limitation-list li,
    .source-list li {
        margin: 0 0 1.1rem 0;
        font-size: 17px;
        line-height: 1.65;
    }

    .limitation-list {
        color: var(--muted);
    }

    blockquote {
        margin: 0.5rem 0 0.75rem 0;
        padding: 0.25rem 0 0.25rem 0.9rem;
        border-left: 2px solid var(--rule);
        color: var(--ink);
        font-size: 15px;
        line-height: 1.55;
    }

    .source-list a {
        color: var(--navy);
        text-decoration-thickness: 1px;
        text-underline-offset: 3px;
    }

    .final-note {
        margin: 2.25rem 0 0 0;
        padding-top: 0.9rem;
        border-top: 1px solid var(--rule);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<h1 class="app-title">Market Research Brief</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="app-subtitle">Four roles work in sequence: orchestrator, researcher, analyst and critic. '
    'Every claim in the brief is tied to a cited source.</p>',
    unsafe_allow_html=True,
)

if "research_running" not in st.session_state:
    st.session_state.research_running = False
if "workflow_state" not in st.session_state:
    st.session_state.workflow_state = None
if "ui_error" not in st.session_state:
    st.session_state.ui_error = None

st.text_input(
    "Company or industry",
    value="Analyze the food delivery market in Nepal.",
    key="research_query",
)
run_clicked = st.button(
    "Run research",
    type="primary",
    disabled=st.session_state.research_running,
)
st.markdown(
    '<p class="run-note">A run typically takes one to three minutes and uses search and language-model credits.</p>',
    unsafe_allow_html=True,
)

ui_error = st.session_state.ui_error
if ui_error:
    kind = ui_error["kind"]
    text = escape(str(ui_error["text"]))
    css_class = "input-error" if kind == "input" else (
        "config-error" if kind == "config" else "failure-notice"
    )
    st.markdown(
        f'<p class="{css_class}">{text}</p>',
        unsafe_allow_html=True,
    )


def stage_index_from_message(message: str) -> int:
    lower = message.lower()
    if "stage 1" in lower or "planning research" in lower or "researching " in lower:
        return 0
    if "stage 2" in lower:
        return 1
    if "stage 3" in lower:
        return 2
    if "stage 4" in lower:
        return 3
    if "mode: resumed" in lower:
        return 1
    return 0


def render_progress(current: int, latest_message: str) -> None:
    stage_names = ["Researcher", "Analyst", "Critic", "Revision"]
    stage_html = []
    for index, name in enumerate(stage_names):
        if index < current:
            classes = "progress-stage finished"
        elif index == current:
            classes = "progress-stage current"
        else:
            classes = "progress-stage"
        stage_html.append(f'<span class="{classes}">{name}</span>')

    progress_placeholder.markdown(
        '<div class="progress-line">' + "".join(stage_html) + "</div>",
        unsafe_allow_html=True,
    )
    message_placeholder.markdown(
        f'<p class="progress-message">{latest_message}</p>',
        unsafe_allow_html=True,
    )


if run_clicked:
    st.session_state.ui_error = None
    query = st.session_state.research_query.strip()
    if not query:
        st.session_state.ui_error = {
            "kind": "input",
            "text": "Enter a company or industry before starting a run.",
        }
    else:
        st.session_state.research_running = True
        st.rerun()

if st.session_state.research_running:
    progress_placeholder = st.empty()
    message_placeholder = st.empty()
    render_progress(0, "Starting research")

    try:
        if API_URL:
            progress_placeholder.markdown(
                '<div class="progress-line"><span class="progress-stage current">Running</span></div>',
                unsafe_allow_html=True,
            )
            message_placeholder.markdown(
                '<p class="progress-message">Running, this can take one to three minutes</p>',
                unsafe_allow_html=True,
            )
            try:
                response = requests.post(
                    f"{API_URL.rstrip('/')}/api/research",
                    json={"query": st.session_state.research_query},
                    timeout=600,
                )
            except requests.Timeout:
                st.session_state.ui_error = {
                    "kind": "failed",
                    "text": "The research API request timed out after ten minutes.",
                }
            except requests.RequestException as error:
                st.session_state.ui_error = {
                    "kind": "failed",
                    "text": f"Could not reach the research API: {error}",
                }
            else:
                if response.status_code >= 400:
                    detail = response.json().get("detail", "The API request failed.")
                    if isinstance(detail, dict):
                        detail = detail.get("message", str(detail))
                    st.session_state.ui_error = {
                        "kind": "config" if response.status_code == 503 else "failed",
                        "text": str(detail),
                    }
                else:
                    workflow_state = WorkflowState.model_validate(response.json()["state"])
                    st.session_state.workflow_state = workflow_state
                    if workflow_state.status == "failed":
                        st.session_state.ui_error = {
                            "kind": "failed",
                            "text": (
                                f"{workflow_state.error or 'The workflow could not complete the run.'} "
                                "Check API keys and credits, or use another provider."
                            ),
                        }
        else:
            def on_progress(message: str) -> None:
                render_progress(stage_index_from_message(message), message)

            try:
                workflow_state = run_pipeline(
                    st.session_state.research_query,
                    on_progress=on_progress,
                )
                st.session_state.workflow_state = workflow_state
                if workflow_state.status == "failed":
                    st.session_state.ui_error = {
                        "kind": "failed",
                        "text": (
                            f"{workflow_state.error or 'The workflow could not complete the run.'} "
                            "Check API keys and credits, or use another provider."
                        ),
                    }
            except ConfigError as error:
                st.session_state.ui_error = {
                    "kind": "config",
                    "text": f"{error}",
                }
    except requests.RequestException as error:
        st.session_state.ui_error = {
            "kind": "failed",
            "text": (
                f"Could not reach the research API: {error}. "
                "Start FastAPI and retry."
            ),
        }
    except Exception as error:
        st.session_state.ui_error = {
            "kind": "failed",
            "text": (
                f"The API response could not be processed: {error}. "
                "Check API keys and credits, or use another provider."
            ),
        }
    finally:
        st.session_state.research_running = False
        st.rerun()


def research_mode_notice(research_mode: str) -> str:
    if research_mode == "resumed":
        return (
            "This brief was produced from saved research data, "
            "not from a new live search."
        )
    return ""


def build_brief_markdown(state) -> str:
    brief = getattr(state, "final_brief", None)
    if not brief:
        return ""

    query = getattr(state, "query", st.session_state.research_query)
    missing = list(getattr(brief, "missing_information", []) or [])
    resumed = getattr(state, "research_mode", "live") == "resumed"
    lines = [
        f"# {brief.subject}",
        "",
        f"*{query}*",
        "",
    ]
    if resumed:
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
            lines.append(f"- **{competitor.name}:** {competitor.description}")
    else:
        lines.append("No competitor is named in the evidence collected.")

    lines.extend(["", "## Key observations", ""])
    if brief.key_observations:
        for observation in brief.key_observations:
            lines.append(f"- {observation.text}")
    else:
        lines.append("No key observations are supported by the evidence collected.")

    lines.extend(["", "## Sources", ""])
    seen_urls = set()
    sources = []
    research_output = getattr(state, "research_output", None)
    if research_output:
        selected_ids = set(brief.source_ids)
        for source in research_output.sources:
            if source.source_id in selected_ids and source.url not in seen_urls:
                seen_urls.add(source.url)
                sources.append(source)
    if sources:
        for source in sources:
            publisher = source.publisher or source.title
            lines.append(
                f"- [{source.title}]({source.url}) - {publisher} "
                f"(accessed {source.accessed_date})"
            )
    else:
        lines.append("- None.")

    lines.extend(["", "## Gaps and limitations", ""])
    if missing:
        for item in missing:
            lines.append(f"- {item}")
    else:
        lines.append("- None.")
    lines.extend([
        "",
        "Claims are checked against the cited sources by rule-based checks "
        "and by a language-model reviewer, but the brief can still contain errors. "
        "Verify anything important.",
    ])
    return "\n".join(lines) + "\n"


def render_brief(state) -> None:
    brief = getattr(state, "final_brief", None)
    if not brief:
        return

    research_output = getattr(state, "research_output", None)
    query = getattr(state, "query", st.session_state.research_query)
    missing = list(getattr(brief, "missing_information", []) or [])

    st.markdown(
        f'<p class="query-line">Query: {escape(str(query))}</p>',
        unsafe_allow_html=True,
    )

    mode = getattr(state, "research_mode", "live")
    mode_notice = research_mode_notice(mode)
    if mode_notice:
        st.markdown(
            f'<p class="mode-notice">{escape(mode_notice)}</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="muted-line">Research mode: Live search</p>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<h2 class="brief-title">{escape(str(brief.subject))}</h2>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="report-section"><h3>Executive summary</h3>', unsafe_allow_html=True)
    st.markdown(escape(str(brief.executive_summary)))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="report-section"><h3>Competitors</h3>', unsafe_allow_html=True)
    if brief.competitors:
        for competitor in brief.competitors:
            st.markdown(
                f'<p><span class="competitor-name">{escape(str(competitor.name))}</span>'
                f' - {escape(str(competitor.description))}</p>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<p class="muted-line">No competitor is named in the evidence collected.</p>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="report-section"><h3>Key observations</h3>', unsafe_allow_html=True)
    if brief.key_observations:
        for observation in brief.key_observations:
            st.markdown(f'<p>{escape(str(observation.text))}</p>', unsafe_allow_html=True)

            findings = []
            if research_output:
                findings = [
                    finding
                    for finding in research_output.findings
                    if finding.finding_id in observation.evidence_finding_ids
                ]

            source_publishers = []
            for finding in findings:
                for source_id in finding.source_ids:
                    source = next(
                        (
                            item
                            for item in research_output.sources
                            if item.source_id == source_id
                        ),
                        None,
                    ) if research_output else None
                    if source:
                        label = source.publisher or source.title
                        if label not in source_publishers:
                            source_publishers.append(label)

            confidence = "unknown"
            if findings:
                confidence = findings[0].confidence
            publisher_text = ", ".join(source_publishers) if source_publishers else "Source not identified"
            st.markdown(
                f'<p class="source-line">Source: {escape(publisher_text)}, '
                f'confidence {escape(str(confidence))}</p>',
                unsafe_allow_html=True,
            )

            with st.expander("Evidence"):
                for finding in findings:
                    st.markdown(
                        f'<blockquote>{escape(str(finding.evidence))}</blockquote>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<p>{escape(str(finding.claim))}</p>',
                        unsafe_allow_html=True,
                    )
                    if finding.scope == "broader_context":
                        st.markdown(
                            '<p class="evidence-label">Broader context</p>',
                            unsafe_allow_html=True,
                        )
    else:
        st.markdown(
            '<p class="muted-line">No key observations are supported by the evidence collected.</p>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="report-section"><h3>Sources</h3>', unsafe_allow_html=True)
    sources = []
    seen_urls = set()
    if research_output:
        selected_ids = set(brief.source_ids)
        for source in research_output.sources:
            if source.source_id in selected_ids and source.url not in seen_urls:
                seen_urls.add(source.url)
                sources.append(source)

    if sources:
        for index, source in enumerate(sources, start=1):
            publisher = source.publisher or source.title
            st.markdown(
                f'{index}. <a href="{escape(str(source.url), quote=True)}">'
                f'{escape(str(source.title))}</a> - {escape(str(publisher))} '
                f'(accessed {escape(str(source.accessed_date))})',
            )
    else:
        st.markdown('<p class="muted-line">No sources are listed.</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="report-section"><h3>Gaps and limitations</h3>', unsafe_allow_html=True)
    if missing:
        for item in missing:
            st.markdown(f'<p class="limitation-list">{escape(str(item))}</p>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<p class="muted-line">No gaps or limitations are identified.</p>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<p class="final-note">Claims are checked against the cited sources by rule-based checks '
        'and by a language-model reviewer, but the brief can still contain errors. '
        'Verify anything important.</p>',
        unsafe_allow_html=True,
    )
    st.download_button(
        "Download brief (Markdown)",
        data=build_brief_markdown(state),
        file_name="market-research-brief.md",
        mime="text/markdown",
        key="download_brief",
    )


workflow_state = st.session_state.workflow_state
if workflow_state is not None:
    render_brief(workflow_state)