"""
ielts_agent.py — Entry point
────────────────────────────
Launches the multi-agent IELTS Writing Tutor (Examiner + Tutor).

Usage
─────
  python ielts_agent.py
  python ielts_agent.py --model ieltstutor --thread student_b22_01
  python ielts_agent.py --examiner-temp 0.1 --tutor-temp 0.7

Package layout
──────────────
  agent/schemas.py      — Pydantic models (IELTSEvaluation, TutorFeedback, …)
  agent/tools.py        — LangChain @tools (count_words, grammar_check, keywords)
  agent/graph.py        — Examiner agent (plan → tools → evaluate → critique)
  agent/tutor_graph.py  — Tutor agent (review → challenge? → lesson plan)
  agent/supervisor.py   — Supervisor StateGraph wiring both agents + HITL
  agent/display.py      — Rich rendering helpers
  agent/cli.py          — IELTSCLI interactive loop
"""

from __future__ import annotations

import argparse
from datetime import datetime

from agent import IELTSCLI


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IELTS Writing Tutor System — Multi-Agent (Examiner + Tutor)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="ieltstutor",
        help="Ollama model tag used by both agents",
    )
    parser.add_argument(
        "--examiner-temp", dest="examiner_temp", default=0.1, type=float,
        help="Examiner sampling temperature (low = consistent scoring)",
    )
    parser.add_argument(
        "--tutor-temp", dest="tutor_temp", default=0.6, type=float,
        help="Tutor sampling temperature (higher = more creative lesson plans)",
    )
    parser.add_argument(
        "--thread",
        default=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Thread ID for checkpoint session isolation",
    )
    parser.add_argument(
        "--postgres-uri",
        default=None,
        help=(
            "PostgreSQL URI for persistent LangGraph memory "
            "(fallback: LANGGRAPH_POSTGRES_URI/POSTGRES_URI/DATABASE_URL)"
        ),
    )
    args = parser.parse_args()

    cli = IELTSCLI(
        model=args.model,
        examiner_temp=args.examiner_temp,
        tutor_temp=args.tutor_temp,
        thread_id=args.thread,
        postgres_uri=args.postgres_uri,
    )
    cli.run()


if __name__ == "__main__":
    main()
