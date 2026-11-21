import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from market_research.agents.researcher import run_researcher

query = " ".join(sys.argv[1:]) or "Analyze the food delivery market in Nepal."
output = run_researcher(query, on_progress=print)

Path("outputs").mkdir(exist_ok=True)
Path("outputs/researcher_output.json").write_text(
    output.model_dump_json(indent=2), encoding="utf-8"
)

print(f"\nQuestions: {len(output.plan.research_questions)}")
print(f"Findings:  {len(output.findings)}")
print(f"Sources:   {len(output.sources)}")
print(f"Missing:   {len(output.missing_information)}")
for item in output.missing_information:
    print(f"  - {item}")
print("\nFull output saved to outputs/researcher_output.json")