# Research Paper Reference: MaRGen

## 1. Paper Overview

**Title:** MaRGen: Multi-Agent LLM Approach for Self-Directed Market Research and Analysis

**Authors:** Roman Koshkin, Pengyu Dai, Nozomi Fujikawa, Masahito Togami, and Marco Visentini-Scarzanella.

**Published:** 2025, LLM4ECommerce Workshop at the 31st ACM SIGKDD Conference.

**Links:**
- [arXiv Paper](https://arxiv.org/abs/2508.01370)
- [Amazon Science Publication](https://www.amazon.science/publications/margen-multi-agent-llm-approach-for-self-directed-market-research-and-analysis)
- [PDF](https://cdn.amazon.science/b4/64/e2ed5d3e41dabf81a641151b89f3/scipub-approval152134-38599640-margen-multiagent-llm-approach-for-self-directed-market-research-and-analysis.pdf)

## 2. What the Paper Explores

MaRGen presents a multi-agent framework designed to automate business analysis and market report generation using Large Language Models (LLMs).

The framework uses four specialized roles:

- **Researcher:** Investigates business questions and performs data analysis.
- **Writer:** Converts research findings into structured reports.
- **Reviewer:** Evaluates generated reports and suggests improvements.
- **Retriever:** Retrieves relevant information from unstructured business documents to support research.

The system also incorporates automated report evaluation and iterative improvement to improve the quality of generated reports.

One of the paper's central ideas is that assigning different responsibilities to specialized agents can help structure a complex research and reporting task.

## 3. Key Takeaways

The following ideas were particularly relevant to this project:

1. **Separation of responsibilities:** Research, report generation, and review are distinct tasks and can be handled by specialized components.
2. **Structured information handoffs:** Research findings should be passed between stages in a consistent format rather than relying entirely on conversational context.
3. **Review and revision:** A dedicated reviewer can identify unsupported claims, missing information, and weaknesses in a generated report.
4. **Iterative improvement:** Feedback from the reviewer can be used to improve the initial report.
5. **Evidence-based reporting:** The quality of a research report depends on the information available to the system, not simply the fluency of the generated text.

## 4. How This Project Uses These Ideas

This project takes inspiration from MaRGen's separation of responsibilities while implementing a smaller workflow suitable for the assignment.

| MaRGen | This Project |
|---|---|
| Researcher | Researcher |
| Retriever | Research and retrieval functionality within the Researcher |
| Writer | Analyst |
| Reviewer | Critic |
| Workflow coordination | Orchestrator |

The implementation uses four main components:

- **Orchestrator:** Controls the workflow, manages shared state, and handles handoffs between components.
- **Researcher:** Internally plans what information to investigate, searches relevant web sources, and collects research findings.
- **Analyst:** Produces a structured research brief based on the collected evidence.
- **Critic:** Reviews the brief for factual support, completeness, and clarity, and can request one revision.

The Research Planner is an internal stage of the Researcher rather than an independent agent.

## 5. Differences from the Original Paper

This project is not intended to reproduce the complete MaRGen framework.

The original paper describes a more extensive system involving business databases, SQL-based analysis, retrieval of domain-specific knowledge, report generation, and automated evaluation.

This implementation instead focuses on a smaller web-based research workflow:

- Uses web research rather than access to internal business databases.
- Uses a lightweight research-planning stage.
- Produces concise research briefs instead of detailed consulting reports.
- Uses a single review-and-revision cycle.
- Uses one configurable LLM provider.

These decisions keep the implementation focused on the assignment's requirements while retaining the core ideas of specialized responsibilities, structured handoffs, and review.

## 6. Conclusion

MaRGen provided a useful architectural reference for designing this application. Its approach helped inform the separation of research, analysis, and review responsibilities.

The project adapts these concepts into a simpler and more constrained implementation, prioritizing clear workflow orchestration, structured data exchange, and evidence-supported research outputs.