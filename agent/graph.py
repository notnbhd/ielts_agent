"""
agent/graph.py
──────────────
LangGraph StateGraph — IELTS evaluation workflow.

Key design: tools are executed DIRECTLY in Python (no bind_tools / Ollama tool API).
This makes the system compatible with any Ollama model regardless of whether it
declares tool support, and is more reliable than letting a small LLM orchestrate calls.

Workflow (eval mode)
────────────────────
  START → auto_tools ──[HITL interrupt]──► evaluate → critique ──► END
                                                           │
                                                    [revision?]──► evaluate

  Chat mode: START → chat ──► END

Features
────────
  1. Structured Output   — evaluate_node uses llm.with_structured_output(IELTSEvaluation)
                           with plain-text fallback for small/quantized models.
  2. Self-Reflection     — critique_node does rule-based consistency checks and
                           triggers a revision pass (capped at MAX_REVISIONS=1).
  3. Human-in-the-Loop  — interrupt_before=["evaluate"] pauses between tools and scoring.

Exports
───────
  AgentState   — TypedDict for use in supervisor
  auto_tools_node, chat_node, evaluate_node, critique_node  — individual nodes
  entry_router, critique_router  — conditional edge functions
  MAX_REVISIONS
  build_graph  — single-agent compiled graph factory
"""

from __future__ import annotations

import json
from typing import Any
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from rich.console import Console

from agent.prompts import SYSTEM_PROMPT
from agent.schemas import IELTSEvaluation
from agent.tools import TOOL_MAP

console = Console()

MAX_REVISIONS = 1


# ══════════════════════════════════════════════════════════════════════════════
# State
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages:          Annotated[list[BaseMessage], add_messages]
    is_eval:           bool   # True → evaluation flow; False → chat
    tool_results:      dict   # raw JSON strings from tool execution
    evaluation:        dict   # IELTSEvaluation.model_dump() or {}
    critique_feedback: str    # non-empty = inconsistencies found
    revision_count:    int    # number of revision passes completed


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_llm(model: str, temperature: float) -> ChatOllama:
    return ChatOllama(model=model, temperature=temperature, num_predict=2048, num_ctx=4096)


def _tool_summary(tool_results: dict) -> str:
    """Human-readable summary of tool outputs injected into the evaluate prompt."""
    parts: list[str] = []
    if "count_words" in tool_results:
        d = json.loads(tool_results["count_words"])
        parts.append(f"• Word Count: {d['word_count']} ({d['status']})")
    if "grammar_check" in tool_results:
        d = json.loads(tool_results["grammar_check"])
        parts.append(f"• Grammar Issues: {d['issues_found']} / {d['total_sentences']} sentences")
    if "topic_keywords" in tool_results:
        d = json.loads(tool_results["topic_keywords"])
        parts.append(
            f"• Lexical Diversity: {d['unique_word_ratio']} — {d['lexical_diversity_note']}"
        )
    return "\n".join(parts)


def _extract_essay(state: AgentState) -> str:
    """Pull the raw essay text out of the last HumanMessage."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and "**Essay:**" in msg.content:
            import re
            m = re.search(r"\*\*Essay:\*\*\s*(.*)", msg.content, re.DOTALL)
            if m:
                return m.group(1).strip()
    # Fallback: return the whole last human message
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# Nodes
# ══════════════════════════════════════════════════════════════════════════════

def auto_tools_node(state: AgentState) -> dict:
    """
    AUTO TOOLS node — runs all three analysis tools directly in Python.

    No bind_tools() / Ollama tool API is used here. Tools are called
    unconditionally whenever is_eval=True, making this compatible with
    any Ollama model regardless of tool support declarations.
    """
    essay = _extract_essay(state)
    tool_results: dict = {}

    for name in ["count_words", "grammar_check", "topic_keywords"]:
        t = TOOL_MAP.get(name)
        if t is None:
            continue
        try:
            result = t.invoke({"text": essay})
            console.print(f"[cyan]🛠️  Tool: [bold]{name}[/bold][/cyan]")
        except Exception as exc:
            result = json.dumps({"error": str(exc)})
            console.print(f"[red]🛠️  Tool {name} failed: {exc}[/red]")
        tool_results[name] = result

    return {"tool_results": tool_results}


def chat_node(state: AgentState, *, model: str, temperature: float) -> dict:
    """
    CHAT node — plain conversation mode, no tools, no evaluation structure.
    The model responds directly to the user's message.
    """
    llm = _make_llm(model, temperature)
    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def evaluate_node(state: AgentState, *, model: str, temperature: float) -> dict:
    """
    EVALUATE node — Structured Output (Feature 1).

    Injects the full tool analysis summary into the prompt, then calls the
    model with with_structured_output(IELTSEvaluation) to produce a validated
    Pydantic JSON response.  Falls back to plain text if structured output fails.
    """
    llm = _make_llm(model, temperature)
    tool_results      = state.get("tool_results", {})
    critique_feedback = state.get("critique_feedback", "")

    summary = _tool_summary(tool_results)
    inject_parts = [
        "Tool analysis summary (incorporate this data into your scores):",
        summary,
    ]
    if critique_feedback:
        inject_parts += [
            "",
            "⚠️  CRITIQUE FEEDBACK (you MUST address these inconsistencies):",
            critique_feedback,
        ]
    inject_parts.append(
        "\nNow produce the full IELTS evaluation in the required structured format."
    )

    inject   = SystemMessage(content="\n".join(inject_parts))
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"] + [inject]

    evaluation_dict: dict = {}
    ai_content: str = ""

    try:
        llm_structured = llm.with_structured_output(IELTSEvaluation)
        ev: IELTSEvaluation = llm_structured.invoke(messages)
        evaluation_dict = ev.model_dump()
        ai_content      = ev.to_markdown()
        console.print("[dim]✅ Structured output parsed successfully[/dim]")
    except Exception as exc:
        console.print(f"[dim yellow]⚠️  Structured output failed ({exc}), using plain text[/dim yellow]")
        response   = llm.invoke(messages)
        ai_content = response.content

    return {
        "messages":         [AIMessage(content=ai_content)],
        "evaluation":       evaluation_dict,
        "critique_feedback": "",   # reset after each pass
    }


def critique_node(state: AgentState) -> dict:
    """
    CRITIQUE node — Self-Reflection (Feature 2).

    Rule-based consistency checker (no LLM call). Compares structured scores
    against objective tool measurements and flags contradictions.
    """
    evaluation   = state.get("evaluation", {})
    tool_results = state.get("tool_results", {})
    issues: list[str] = []

    if not evaluation:
        return {"critique_feedback": ""}

    ta_score  = evaluation.get("task_achievement",  {}).get("score", 0)
    gra_score = evaluation.get("grammatical_range", {}).get("score", 0)
    lr_score  = evaluation.get("lexical_resource",  {}).get("score", 0)
    cc_score  = evaluation.get("coherence_cohesion", {}).get("score", 0)
    overall   = evaluation.get("overall_band", 0)
    sub_mean  = (ta_score + gra_score + lr_score + cc_score) / 4

    if "count_words" in tool_results:
        d = json.loads(tool_results["count_words"])
        if d["word_count"] < 250 and ta_score >= 7.0:
            issues.append(
                f"TA is {ta_score} but essay has only {d['word_count']} words "
                "(below 250-word minimum). TA should be reduced."
            )

    if "grammar_check" in tool_results:
        d = json.loads(tool_results["grammar_check"])
        if d["issues_found"] > 4 and gra_score >= 7.0:
            issues.append(
                f"GRA is {gra_score} but grammar tool flagged "
                f"{d['issues_found']} issues. GRA should be reduced."
            )

    if "topic_keywords" in tool_results:
        d = json.loads(tool_results["topic_keywords"])
        if d["unique_word_ratio"] < 0.45 and lr_score >= 7.0:
            issues.append(
                f"LR is {lr_score} but unique-word ratio is only "
                f"{d['unique_word_ratio']} (poor diversity). LR should be reduced."
            )

    if abs(overall - sub_mean) > 0.75:
        issues.append(
            f"Overall band ({overall}) deviates from sub-score mean "
            f"({sub_mean:.2f}) by more than 0.75."
        )

    if issues:
        feedback = "Critique found inconsistencies:\n" + "\n".join(
            f"  {i+1}. {issue}" for i, issue in enumerate(issues)
        )
        console.print(f"[yellow]🔍 Critique: {len(issues)} issue(s) → revision[/yellow]")
        return {
            "critique_feedback": feedback,
            "revision_count":    state.get("revision_count", 0) + 1,
        }

    console.print("[dim green]✅ Critique: scores are consistent[/dim green]")
    return {"critique_feedback": ""}


# ══════════════════════════════════════════════════════════════════════════════
# Routers
# ══════════════════════════════════════════════════════════════════════════════

def entry_router(state: AgentState) -> Literal["auto_tools", "chat"]:
    """START → auto_tools (eval) or chat (conversation)."""
    return "auto_tools" if state["is_eval"] else "chat"


def critique_router(state: AgentState) -> Literal["evaluate", "__end__"]:
    """After critique: re-evaluate if needed (up to MAX_REVISIONS), else end."""
    has_issues = bool(state.get("critique_feedback", ""))
    under_cap  = state.get("revision_count", 0) <= MAX_REVISIONS
    if has_issues and under_cap:
        console.print(
            f"[dim yellow]↩️  Revision {state['revision_count']}/{MAX_REVISIONS}[/dim yellow]"
        )
        return "evaluate"
    return "__end__"


# ══════════════════════════════════════════════════════════════════════════════
# Graph factory (single-agent, kept for backward compatibility)
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(model: str, temperature: float, checkpointer: Any | None = None):
    """Single-agent graph (Examiner only). Use build_supervisor_graph for multi-agent."""
    builder = StateGraph(AgentState)

    def _chat(s):
        return chat_node(s, model=model, temperature=temperature)

    def _evaluate(s):
        return evaluate_node(s, model=model, temperature=temperature)

    builder.add_node("auto_tools", auto_tools_node)
    builder.add_node("chat",       _chat)
    builder.add_node("evaluate",   _evaluate)
    builder.add_node("critique",   critique_node)

    builder.add_conditional_edges(START, entry_router, {
        "auto_tools": "auto_tools",
        "chat":       "chat",
    })
    builder.add_edge("auto_tools", "evaluate")
    builder.add_edge("chat",       END)
    builder.add_edge("evaluate",   "critique")
    builder.add_conditional_edges("critique", critique_router, {
        "evaluate": "evaluate",
        "__end__":  END,
    })

    if checkpointer is None:
        checkpointer = MemorySaver()

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["evaluate"],
    )
