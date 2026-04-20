"""
agent/supervisor.py
───────────────────
Multi-agent supervisor — Examiner Agent + Tutor Agent.

No bind_tools() / Ollama tool API is used anywhere. Tools run directly in
Python via auto_tools_node, so the system works with any Ollama model.

Workflow
────────
  START
   │
   ├──[is_eval=False]──► chat ──► END
   │
   └──[is_eval=True]───► auto_tools
                              │
                       [HITL interrupt]
                              │
                          evaluate ──► critique
                              ↑            │
                              └──[revise?]─┘
                                           │ done
                                     tutor_review
                                          │
                              ┌──[challenge?]──┐
                              │                │
                    examiner_reconsider   tutor_lesson_plan
                              │
                        tutor_lesson_plan ──► END

Exports
───────
  SupervisorState       — combined TypedDict
  build_supervisor_graph — compiled graph factory
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from rich.console import Console

# Examiner nodes and routers
from agent.graph import (
    MAX_REVISIONS,
    auto_tools_node,
    chat_node,
    critique_node,
    critique_router,
    entry_router,
    evaluate_node,
)

# Tutor nodes and router
from agent.tutor_graph import (
    challenge_router,
    examiner_reconsider_node,
    tutor_lesson_plan_node,
    tutor_review_node,
)

console = Console()


# ══════════════════════════════════════════════════════════════════════════════
# Supervisor State
# ══════════════════════════════════════════════════════════════════════════════

class SupervisorState(TypedDict):
    # Shared / Examiner fields
    messages:          Annotated[list[BaseMessage], add_messages]
    is_eval:           bool
    tool_results:      dict
    evaluation:        dict
    critique_feedback: str
    revision_count:    int
    # Tutor fields
    tutor_challenge:   str   # serialised list[ChallengeSignal] JSON or ""
    examiner_verdict:  str   # "revised" | "held" | ""
    tutor_feedback:    dict  # TutorFeedback.model_dump() or {}


# ══════════════════════════════════════════════════════════════════════════════
# Graph factory
# ══════════════════════════════════════════════════════════════════════════════

def build_supervisor_graph(
    model: str,
    examiner_temp: float = 0.1,
    tutor_temp: float    = 0.6,
):
    """
    Compile and return the multi-agent supervisor graph.

    Parameters
    ----------
    model         : Ollama model tag used by BOTH agents
    examiner_temp : Low temperature for consistent, objective scoring
    tutor_temp    : Higher temperature for creative, personalised lesson plans
    """
    builder = StateGraph(SupervisorState)

    # ── Partial-bind helpers ──────────────────────────────────────────────────

    def _chat(s):
        return chat_node(s, model=model, temperature=examiner_temp)

    def _evaluate(s):
        return evaluate_node(s, model=model, temperature=examiner_temp)

    def _examiner_reconsider(s):
        return examiner_reconsider_node(s, model=model, temperature=examiner_temp)

    def _tutor_lesson_plan(s):
        return tutor_lesson_plan_node(s, model=model, temperature=tutor_temp)

    # ── Examiner nodes ────────────────────────────────────────────────────────
    builder.add_node("auto_tools", auto_tools_node)
    builder.add_node("chat",       _chat)
    builder.add_node("evaluate",   _evaluate)
    builder.add_node("critique",   critique_node)

    # ── Tutor nodes ───────────────────────────────────────────────────────────
    builder.add_node("tutor_review",        tutor_review_node)
    builder.add_node("examiner_reconsider", _examiner_reconsider)
    builder.add_node("tutor_lesson_plan",   _tutor_lesson_plan)

    # ── Edges: entry ──────────────────────────────────────────────────────────
    builder.add_conditional_edges(START, entry_router, {
        "auto_tools": "auto_tools",
        "chat":       "chat",
    })
    builder.add_edge("chat", END)

    # ── Edges: Examiner subgraph ───────────────────────────────────────────────
    builder.add_edge("auto_tools", "evaluate")      # tools always go straight to evaluate
    builder.add_edge("evaluate",   "critique")
    builder.add_conditional_edges("critique", critique_router, {
        "evaluate": "evaluate",        # self-revision loop
        "__end__":  "tutor_review",    # done → hand off to Tutor
    })

    # ── Edges: Tutor subgraph ─────────────────────────────────────────────────
    builder.add_conditional_edges("tutor_review", challenge_router, {
        "examiner_reconsider": "examiner_reconsider",
        "tutor_lesson_plan":   "tutor_lesson_plan",
    })
    builder.add_edge("examiner_reconsider", "tutor_lesson_plan")
    builder.add_edge("tutor_lesson_plan",   END)

    return builder.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["evaluate"],   # HITL: user reviews tool results before scoring
    )
