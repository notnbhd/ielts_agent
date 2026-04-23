"""
agent/display.py
────────────────
Score extraction and Rich table/panel rendering utilities.

Handles both output paths:
  • Structured path  — evaluation dict from IELTSEvaluation.model_dump()
  • Fallback path    — regex-parsed scores from plain LLM text

Exports
───────
  parse_scores            — regex extractor (fallback only)
  display_score_table     — coloured band-score table
  display_tool_summary    — human-readable tool results panel (for HITL review)
  display_evaluation      — full Pydantic evaluation panel with per-criterion detail
"""

from __future__ import annotations

import json
import re
from typing import Any

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()

# ── Constants ──────────────────────────────────────────────────────────────────

_LEVEL_MAP: dict[int, str] = {
    9: "Expert",
    8: "Very Good",
    7: "Good",
    6: "Competent",
    5: "Modest",
    4: "Limited",
    0: "Below Band 4",
}

_CRITERION_NAMES: dict[str, str] = {
    "Overall": "🎯 Overall Band Score",
    "TA":      "📝 Task Achievement",
    "CC":      "🔗 Coherence & Cohesion",
    "LR":      "📚 Lexical Resource",
    "GRA":     "✏️  Grammatical Range & Accuracy",
}

_SCORE_PATTERNS: dict[str, str] = {
    "Overall": r"Band Score:\s*([0-9.]+)",
    "TA":      r"Task Achievement.*?:\s*([0-9.]+)",
    "CC":      r"Coherence and Cohesion.*?:\s*([0-9.]+)",
    "LR":      r"Lexical Resource.*?:\s*([0-9.]+)",
    "GRA":     r"Grammatical Range.*?:\s*([0-9.]+)",
}


# ── Private helpers ────────────────────────────────────────────────────────────

def _get_level(score: float) -> str:
    for threshold in sorted(_LEVEL_MAP, reverse=True):
        if score >= threshold:
            return _LEVEL_MAP[threshold]
    return "Below Band 4"


def _score_color(score: float) -> str:
    return "green" if score >= 7 else "yellow" if score >= 5.5 else "red"


# ── Public: fallback score extraction ─────────────────────────────────────────

def parse_scores(response: str) -> dict[str, float]:
    """Extract IELTS band scores from a raw LLM text response (fallback path)."""
    scores: dict[str, float] = {}
    for key, pattern in _SCORE_PATTERNS.items():
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                scores[key] = float(match.group(1))
            except ValueError:
                pass
    return scores


# ── Public: score table (used by both paths) ───────────────────────────────────

def display_score_table(scores: dict[str, float]) -> None:
    """Render a coloured band-score summary table from any scores dict."""
    if not scores:
        console.print("[yellow]No scores found.[/yellow]")
        return

    table = Table(
        title="📊 IELTS Band Scores",
        box=box.ROUNDED,
        style="cyan",
        title_style="bold cyan",
    )
    table.add_column("Criterion", style="bold white", width=36)
    table.add_column("Score",     justify="center", style="bold yellow", width=8)
    table.add_column("Level",     justify="center", width=18)

    for key in ["Overall", "TA", "CC", "LR", "GRA"]:
        if key not in scores:
            continue
        score = scores[key]
        color = _score_color(score)
        table.add_row(
            _CRITERION_NAMES[key],
            f"[{color}]{score}[/{color}]",
            f"[{color}]{_get_level(score)}[/{color}]",
        )

    console.print()
    console.print(table)
    console.print()


# ── Public: Human-in-the-Loop tool summary panel ──────────────────────────────

def display_tool_summary(tool_results: dict) -> None:
    """
    Display a formatted panel showing tool analysis results.
    Called by the CLI during the HITL review step so the user can see what
    the tools found before approving the evaluation.
    """
    lines: list[str] = []

    if "count_words" in tool_results:
        d = json.loads(tool_results["count_words"])
        status_color = "green" if d["word_count"] >= 250 else "yellow"
        lines.append(
            f"[bold]Word Count:[/bold] [{status_color}]{d['word_count']}[/{status_color}]"
            f"  {d['status']}"
        )

    if "grammar_check" in tool_results:
        d = json.loads(tool_results["grammar_check"])
        n = d["issues_found"]
        color = "green" if n == 0 else "yellow" if n <= 3 else "red"
        lines.append(
            f"[bold]Grammar Issues:[/bold] [{color}]{n}[/{color}]"
            f" / {d['total_sentences']} sentences"
        )
        for issue in d.get("details", []):
            lines.append(f"  [dim]• Sentence {issue['sentence']}: {issue['issue']}[/dim]")

    if "topic_keywords" in tool_results:
        d = json.loads(tool_results["topic_keywords"])
        ratio = d["unique_word_ratio"]
        color = "green" if ratio > 0.6 else "yellow"
        top_kw = ", ".join(list(d["top_keywords"].keys())[:8])
        lines.append(
            f"[bold]Lexical Diversity:[/bold] [{color}]{ratio}[/{color}]"
            f" — {d['lexical_diversity_note']}"
        )
        lines.append(f"  [dim]Top keywords: {top_kw}[/dim]")

    console.print(Panel(
        "\n".join(lines),
        title="🔬 Tool Analysis Report",
        title_align="left",
        box=box.ROUNDED,
        border_style="cyan",
        padding=(1, 2),
    ))


# ── Public: full structured evaluation panel ──────────────────────────────────

def display_evaluation(evaluation_dict: dict) -> None:
    """
    Render a full Pydantic-backed structured evaluation using Rich Markdown.
    Used when structured output succeeded.
    """
    if not evaluation_dict:
        return
    # Reconstruct to_markdown() logic from the dict without re-importing schema
    ta  = evaluation_dict.get("task_achievement",  {})
    cc  = evaluation_dict.get("coherence_cohesion", {})
    lr  = evaluation_dict.get("lexical_resource",   {})
    gra = evaluation_dict.get("grammatical_range",  {})

    md_lines = [
        f"## Band Score: {evaluation_dict.get('overall_band', '?')}",
        "",
        f"### Task Achievement (TA): {ta.get('score', '?')}",
        ta.get("analysis", ""),
        "",
        f"### Coherence and Cohesion (CC): {cc.get('score', '?')}",
        cc.get("analysis", ""),
        "",
        f"### Lexical Resource (LR): {lr.get('score', '?')}",
        lr.get("analysis", ""),
        "",
        f"### Grammatical Range and Accuracy (GRA): {gra.get('score', '?')}",
        gra.get("analysis", ""),
        "",
        "### Key Strengths",
        *[f"- {s}" for s in evaluation_dict.get("key_strengths", [])],
        "",
        "### Areas for Improvement",
        *[f"- {a}" for a in evaluation_dict.get("areas_for_improvement", [])],
    ]
    rewrites = evaluation_dict.get("rewrite_suggestions", [])
    if rewrites:
        md_lines += ["", "### Rewrite Suggestions", *[f"- {r}" for r in rewrites]]

    console.print(Markdown("\n".join(md_lines)))


# ── Public: Tutor lesson plan panel ───────────────────────────────────────────

def display_tutor_feedback(tutor_feedback: dict) -> None:
    """
    Render the Tutor Agent's TutorFeedback as a structured Rich panel.
    Falls back gracefully if tutor_feedback is empty.
    """
    if not tutor_feedback:
        return

    focus      = tutor_feedback.get("priority_focus", "")
    encourage  = tutor_feedback.get("encouragement", "")
    plan       = tutor_feedback.get("lesson_plan", [])
    tips       = tutor_feedback.get("next_essay_tips", [])
    rewrites   = tutor_feedback.get("rewrite_examples", [])

    focus_color = {"TA": "blue", "CC": "magenta", "LR": "green", "GRA": "yellow"}.get(focus, "cyan")

    lines: list[str] = [
        f"[bold]Priority Focus:[/bold] [{focus_color}]{focus}[/{focus_color}]\n",
        f"[italic]{encourage}[/italic]\n",
        "[bold]📋 Your Improvement Plan:[/bold]",
        *[f"  {i+1}. {step}" for i, step in enumerate(plan)],
        "",
        "[bold]🎯 For Your Next Essay:[/bold]",
        *[f"  • {tip}" for tip in tips],
    ]
    if rewrites:
        lines += ["", "[bold]✏️  Rewrite Examples:[/bold]", *[f"  {r}" for r in rewrites]]

    console.print()
    console.print(Panel(
        "\n".join(lines),
        title="🎓 Tutor's Personalised Lesson Plan",
        title_align="left",
        box=box.DOUBLE,
        border_style="green",
        padding=(1, 2),
    ))
    console.print()


def display_tutor_react_debug(react_steps: list[dict[str, Any]], react_meta: dict[str, Any] | None = None) -> None:
    """Render Tutor ReAct trace and timing metadata for debugging/audit."""
    meta = react_meta or {}

    if not react_steps and not meta:
        console.print("[yellow]No Tutor ReAct trace available yet.[/yellow]")
        return

    header_lines = [
        f"[bold]Loop limit:[/bold] {meta.get('loop_limit', '?')}",
        f"[bold]Steps:[/bold] {meta.get('steps_count', len(react_steps))}",
        f"[bold]ReAct duration:[/bold] {meta.get('react_duration_ms', 0)} ms",
        f"[bold]Formatter duration:[/bold] {meta.get('formatter_duration_ms', 0)} ms",
        f"[bold]Total duration:[/bold] {meta.get('total_duration_ms', 0)} ms",
        f"[bold]Used fallback:[/bold] {meta.get('used_fallback', False)}",
    ]
    react_error = meta.get("react_error", "")
    if react_error:
        header_lines.append(f"[bold]ReAct error:[/bold] {react_error}")

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("#", justify="right", width=4)
    table.add_column("Source", width=9)
    table.add_column("Type", width=12)
    table.add_column("Duration", justify="right", width=10)
    table.add_column("Detail", overflow="fold")

    for step in react_steps:
        step_type = str(step.get("step_type", "?"))
        source = str(step.get("source", "?"))
        duration = step.get("duration_ms", 0)
        if isinstance(duration, (int, float)):
            duration_text = f"{duration:.2f} ms"
        else:
            duration_text = "-"

        if step_type == "tool_call":
            detail = f"{step.get('tool', '?')} args={step.get('args', {})}"
        elif step_type == "tool_result":
            detail = f"{step.get('tool', '?')} -> {str(step.get('content', ''))[:180]}"
        else:
            detail = str(step.get("content", ""))[:180]

        table.add_row(
            str(step.get("step_index", "?")),
            source,
            step_type,
            duration_text,
            detail,
        )

    console.print()
    console.print(Panel(
        "\n".join(header_lines),
        title="Tutor ReAct Meta",
        title_align="left",
        border_style="cyan",
        box=box.ROUNDED,
    ))
    console.print(table)
    console.print()
