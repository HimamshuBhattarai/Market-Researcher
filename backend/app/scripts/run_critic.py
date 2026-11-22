import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from market_research.agents.critic import run_critic
from market_research.schemas import ResearchBrief, ResearcherOutput

research = ResearcherOutput.model_validate_json(
    Path("outputs/fixture_nepal.json").read_text(encoding="utf-8")
)
brief = ResearchBrief.model_validate_json(
    Path("outputs/analyst_brief.json").read_text(encoding="utf-8")
)

fb = run_critic("Analyze the food delivery market in Nepal.", research, brief)

print("APPROVED:", fb.approved)
print("ASSESSMENT:", fb.overall_assessment)
print("\nISSUES")
for i in fb.issues:
    print("-", i)
print("\nREVISION INSTRUCTIONS")
for r in fb.revision_instructions:
    print("-", r)

Path("outputs").mkdir(exist_ok=True)
Path("outputs/critic_feedback.json").write_text(
    fb.model_dump_json(indent=2), encoding="utf-8"
)
print("\nSaved outputs/critic_feedback.json")