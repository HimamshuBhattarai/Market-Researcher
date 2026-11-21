from pydantic import Field

from market_research.config import MAX_QUESTIONS
from market_research.schemas import ResearchPlan, ResearchQuestion
from market_research.schemas.base import StrictModel
from market_research.services.llm import LLMError, generate_structured

DIMENSIONS = [
    "Market size",
    "Market growth",
    "Competition",
    "Business models",
    "Customers",
    "Economics",
    "Regulation",
    "Challenges",
    "Opportunities",
]

METRIC_HINTS = [
    "Total addressable market (TAM)",
    "Market growth rate",
    "Number of active users",
    "Average order value (AOV)",
    "Order frequency",
    "Customer acquisition cost (CAC)",
    "Customer lifetime value (LTV)",
    "Delivery cost per order",
    "Commission rate",
    "Customer retention",
]

SYSTEM = (
    "You are a research planner. You decide WHAT must be investigated. "
    "You never state facts, figures or conclusions about the topic."
)


class _QuestionDraft(StrictModel):
    question: str
    research_dimension: str
    relevant_metrics: list[str] = Field(default_factory=list)


class _PlanDraft(StrictModel):
    subject: str
    domain: str
    geography: str | None = None
    research_questions: list[_QuestionDraft]
    required_metrics: list[str] = Field(default_factory=list)


def make_plan(query: str) -> ResearchPlan:
    prompt = f"""User research request: "{query}"

Produce a research plan.

Rules:
- subject: the company, industry or market being researched.
- domain: the business domain (e.g. "food delivery").
- geography: the country/region, or null if none applies.
- research_questions: at most {MAX_QUESTIONS} questions in total, at most one per
  research dimension. Pick the dimensions most useful for a short market brief.
  Each question must be answerable from public web sources.
- research_dimension: must be exactly one of: {", ".join(DIMENSIONS)}.
- relevant_metrics: metrics for that question, chosen only from this list
  (use an empty list if none apply): {"; ".join(METRIC_HINTS)}.
- required_metrics: the overall metrics the brief should ideally cover, from the same list.
- Do NOT answer the questions or assume any conclusions."""

    draft = generate_structured(prompt, _PlanDraft, system=SYSTEM)

    drafts = draft.research_questions[:MAX_QUESTIONS]
    if not drafts:
        raise LLMError("Planner returned no research questions.")

    questions = [
        ResearchQuestion(
            question_id=f"q{i}",
            question=d.question,
            research_dimension=d.research_dimension,
            relevant_metrics=d.relevant_metrics,
        )
        for i, d in enumerate(drafts, start=1)
    ]
    dimensions = list(dict.fromkeys(q.research_dimension for q in questions))

    return ResearchPlan(
        subject=draft.subject,
        domain=draft.domain,
        geography=draft.geography,
        research_dimensions=dimensions,
        research_questions=questions,
        required_metrics=draft.required_metrics,
    )