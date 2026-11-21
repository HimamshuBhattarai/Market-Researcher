import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from market_research.agents.analyst import run_analyst
from market_research.schemas import ResearcherOutput

path = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/researcher_output.json")
research = ResearcherOutput.model_validate_json(path.read_text(encoding="utf-8"))

brief = run_analyst("Analyze the food delivery market in Nepal.", research)

print("\nSUMMARY\n", brief.executive_summary)
print("\nCOMPETITORS")
for c in brief.competitors:
    print(f"- {c.name}: {c.description} {c.evidence_finding_ids}")
print("\nOBSERVATIONS")
for o in brief.key_observations:
    print(f"- {o.text} {o.evidence_finding_ids}")
print("\nSOURCES:", brief.source_ids)
print("\nMISSING")
for m in brief.missing_information:
    print(f"- {m}")

Path("outputs").mkdir(exist_ok=True)
Path("outputs/analyst_brief.json").write_text(
    brief.model_dump_json(indent=2), encoding="utf-8"
)