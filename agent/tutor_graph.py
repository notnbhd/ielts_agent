"""
agent/tutor_graph.py
────────────────────
Tutor Agent nodes for the multi-agent supervisor.

The Tutor Agent has two responsibilities:

  1. Challenge Review  — rule-based scan of the Examiner's scores vs. essay
                         evidence.  Raises ChallengeSignals when a score looks
                         unjustifiably low given what the student demonstrably
                         attempted (without requiring an extra LLM call).

  2. Lesson Planning   — uses llm.with_structured_output(TutorFeedback) with a
                         slightly higher temperature to produce personalised,
                         creative improvement guidance.

Exported nodes (called by supervisor.py)
─────────────────────────────────────────
  tutor_review_node       — produces tutor_challenge (list serialised to str)
  examiner_reconsider_node— examiner considers challenge, returns ExaminerVerdict
  tutor_lesson_plan_node  — generates final TutorFeedback for the student

Exported routers
────────────────
  challenge_router        — routes to reconsider or lesson_plan after review
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from rich.console import Console

from agent.schemas import (
    ChallengeSignal,
    ExaminerVerdict,
    IELTSEvaluation,
    TutorFeedback,
)

if TYPE_CHECKING:
    pass   # avoid circular import at type-check time

console = Console()


# ══════════════════════════════════════════════════════════════════════════════
# Tutor system prompts
# ══════════════════════════════════════════════════════════════════════════════

_TUTOR_SYSTEM = """\
You are a compassionate and experienced IELTS Writing tutor.
Your role is NOT to score — that is the Examiner's job.
Your role is to help the student understand their results and plan actionable improvement steps.
Always be encouraging, specific, and growth-focused.
Reference concrete details from the essay in your feedback.
"""

_EXAMINER_RECONSIDER_SYSTEM = """\
You are an impartial IELTS Writing examiner reviewing a pedagogical challenge
raised by the student's tutor. The tutor has identified essay evidence that may
warrant a score adjustment.

Your task: weigh the evidence objectively against the official IELTS descriptors.
- If the evidence is valid and the score was too strict → revise it (be fair).
- If the score is correct despite the evidence → maintain it with a clear justification.

Do NOT inflate scores to be kind. Accuracy is paramount.
"""


# ══════════════════════════════════════════════════════════════════════════════
# Rule-based challenge detection (no LLM call — fast + reliable)
# ══════════════════════════════════════════════════════════════════════════════

_THESIS_PATTERNS = [
    r"\b(i believe|in my opinion|i argue|i think|i contend)\b",
    r"\bthis essay (will|argues|discusses|examines)\b",
    r"\b(to (sum up|conclude|summarise)|in conclusion)\b",
]

_DISCOURSE_MARKERS = [
    "furthermore", "moreover", "however", "in contrast", "consequently",
    "therefore", "additionally", "in conclusion", "to begin with", "firstly",
    "secondly", "finally", "in addition", "as a result", "nevertheless",
    "on the other hand", "for instance", "for example", "in other words",
]

_ACADEMIC_WORDS = [
    "furthermore", "consequently", "significant", "substantial", "implement",
    "contribute", "establish", "demonstrate", "facilitate", "indicate",
    "approximately", "fundamental", "perspective", "acknowledge", "analyse",
    "evaluate", "justify", "emphasise", "consideration", "alternative",
]


def _detect_challenges(essay: str, evaluation: dict) -> list[ChallengeSignal]:
    """
    Produce a list of ChallengeSignals where essay evidence contradicts
    a seemingly low Examiner score.  All checks are rule-based.
    """
    challenges: list[ChallengeSignal] = []
    essay_lower = essay.lower()

    ta_score = evaluation.get("task_achievement",  {}).get("score", 0.0)
    cc_score = evaluation.get("coherence_cohesion", {}).get("score", 0.0)
    lr_score = evaluation.get("lexical_resource",   {}).get("score", 0.0)

    # ── TA challenge ─────────────────────────────────────────────────────────
    has_position = any(re.search(p, essay_lower) for p in _THESIS_PATTERNS)
    paragraphs   = [p for p in re.split(r"\n\s*\n", essay.strip()) if p.strip()]
    has_structure = len(paragraphs) >= 3

    if ta_score < 6.0 and has_position and has_structure:
        challenges.append(ChallengeSignal(
            criterion="TA",
            current_score=ta_score,
            suggested_minimum=ta_score + 0.5,
            evidence=(
                f"Essay has a clear personal position and {len(paragraphs)} paragraphs, "
                "indicating deliberate task response structure."
            ),
        ))

    # ── CC challenge ─────────────────────────────────────────────────────────
    found_markers = [m for m in _DISCOURSE_MARKERS if m in essay_lower]
    if cc_score < 5.5 and len(found_markers) >= 4:
        challenges.append(ChallengeSignal(
            criterion="CC",
            current_score=cc_score,
            suggested_minimum=cc_score + 0.5,
            evidence=(
                f"Essay uses {len(found_markers)} cohesive devices: "
                f"{', '.join(found_markers[:6])}."
            ),
        ))

    # ── LR challenge ─────────────────────────────────────────────────────────
    found_academic = [w for w in _ACADEMIC_WORDS if w in essay_lower]
    if lr_score < 5.5 and len(found_academic) >= 5:
        challenges.append(ChallengeSignal(
            criterion="LR",
            current_score=lr_score,
            suggested_minimum=lr_score + 0.5,
            evidence=(
                f"Essay contains {len(found_academic)} academic/formal words: "
                f"{', '.join(found_academic[:6])}."
            ),
        ))

    return challenges


# ══════════════════════════════════════════════════════════════════════════════
# Tutor nodes
# ══════════════════════════════════════════════════════════════════════════════

def tutor_review_node(state: dict) -> dict:
    """
    TUTOR REVIEW node — checks Examiner scores against essay evidence.
    Produces a serialised list of ChallengeSignal dicts stored in
    state["tutor_challenge"].  Empty string = no challenge.
    """
    evaluation = state.get("evaluation", {})
    essay_msg  = next(
        (m for m in state["messages"] if isinstance(m, HumanMessage) and "**Essay:**" in m.content),
        None,
    )
    essay_text = ""
    if essay_msg:
        match = re.search(r"\*\*Essay:\*\*\s*(.*)", essay_msg.content, re.DOTALL)
        if match:
            essay_text = match.group(1).strip()

    if not evaluation or not essay_text:
        console.print("[dim]🎓 Tutor review: no evaluation or essay found — skipping challenges[/dim]")
        return {"tutor_challenge": ""}

    challenges = _detect_challenges(essay_text, evaluation)

    if challenges:
        payload = json.dumps([c.model_dump() for c in challenges], ensure_ascii=False)
        console.print(
            f"[yellow]🎓 Tutor raises [bold]{len(challenges)}[/bold] challenge(s) "
            f"against the Examiner: "
            f"{', '.join(c.criterion for c in challenges)}[/yellow]"
        )
        return {"tutor_challenge": payload}

    console.print("[dim green]🎓 Tutor review: no challenges — scores look fair[/dim green]")
    return {"tutor_challenge": ""}


def examiner_reconsider_node(state: dict, *, model: str, temperature: float) -> dict:
    """
    EXAMINER RECONSIDER node — the Examiner reads the Tutor's challenge and
    produces an ExaminerVerdict (structured output): either revises scores
    or holds firm with a clear rubric-based justification.

    If the Examiner revises, the evaluation dict is updated in state so the
    Tutor's lesson plan reflects the final agreed scores.
    """
    llm = ChatOllama(model=model, temperature=0.1, num_predict=1024, num_ctx=4096)
    evaluation    = state.get("evaluation", {})
    challenges_raw = state.get("tutor_challenge", "")

    try:
        challenges = [ChallengeSignal(**c) for c in json.loads(challenges_raw)]
    except Exception:
        return {"tutor_challenge": ""}   # malformed — skip

    # Build the challenge summary message
    challenge_text = "\n".join(
        f"• [{c.criterion}] Current: {c.current_score} | Suggested floor: {c.suggested_minimum}\n"
        f"  Evidence: {c.evidence}"
        for c in challenges
    )

    msg = HumanMessage(content=(
        "The student's Tutor has raised the following challenges against your scores:\n\n"
        f"{challenge_text}\n\n"
        "Current evaluation summary:\n"
        f"  TA={evaluation.get('task_achievement', {}).get('score', '?')}, "
        f"CC={evaluation.get('coherence_cohesion', {}).get('score', '?')}, "
        f"LR={evaluation.get('lexical_resource', {}).get('score', '?')}, "
        f"GRA={evaluation.get('grammatical_range', {}).get('score', '?')}\n\n"
        "Please review the evidence and return your verdict."
    ))

    try:
        llm_structured = llm.with_structured_output(ExaminerVerdict)
        verdict: ExaminerVerdict = llm_structured.invoke(
            [SystemMessage(content=_EXAMINER_RECONSIDER_SYSTEM), msg]
        )
    except Exception as exc:
        console.print(f"[dim yellow]Examiner reconsider fallback: {exc}[/dim yellow]")
        return {"tutor_challenge": "", "examiner_verdict": "held"}

    updated_eval = dict(evaluation)

    if not verdict.maintains_scores:
        console.print(
            "[cyan]⚖️  Examiner [bold]revised[/bold] scores after Tutor challenge[/cyan]"
        )
        # Apply overrides to the evaluation dict
        field_map = {
            "revised_ta":      ("task_achievement",  "score"),
            "revised_cc":      ("coherence_cohesion", "score"),
            "revised_lr":      ("lexical_resource",   "score"),
            "revised_gra":     ("grammatical_range",  "score"),
        }
        for attr, (criterion_key, score_key) in field_map.items():
            new_val = getattr(verdict, attr, None)
            if new_val is not None and criterion_key in updated_eval:
                updated_eval[criterion_key] = dict(updated_eval[criterion_key])
                updated_eval[criterion_key][score_key] = new_val
        if verdict.revised_overall is not None:
            updated_eval["overall_band"] = verdict.revised_overall
    else:
        console.print(
            "[dim]⚖️  Examiner [bold]held[/bold] original scores — "
            f"{verdict.justification[:80]}…[/dim]"
        )

    verdict_note = AIMessage(
        content=(
            f"**Examiner's response to Tutor challenge:**\n"
            f"{'✅ Scores revised' if not verdict.maintains_scores else '🔒 Scores maintained'}\n"
            f"_{verdict.justification}_"
        )
    )
    return {
        "messages":       [verdict_note],
        "evaluation":     updated_eval,
        "tutor_challenge": "",       # consumed
        "examiner_verdict": "revised" if not verdict.maintains_scores else "held",
    }


def tutor_lesson_plan_node(state: dict, *, model: str, temperature: float) -> dict:
    """
    TUTOR LESSON PLAN node — generates the final pedagogical TutorFeedback.

    Uses the final (possibly revised) Examiner scores to determine the
    priority focus, then produces an encouraging, personalised improvement plan.
    Higher temperature (0.6) encourages creative, varied language.
    """
    llm = ChatOllama(model=model, temperature=temperature, num_predict=1536, num_ctx=4096)
    evaluation = state.get("evaluation", {})

    # Determine weakest criterion for priority_focus
    criterion_scores = {
        "TA":  evaluation.get("task_achievement",  {}).get("score", 0.0),
        "CC":  evaluation.get("coherence_cohesion", {}).get("score", 0.0),
        "LR":  evaluation.get("lexical_resource",   {}).get("score", 0.0),
        "GRA": evaluation.get("grammatical_range",  {}).get("score", 0.0),
    }
    weakest = min(criterion_scores, key=criterion_scores.get)

    # Build context from the evaluation
    eval_summary = (
        f"TA={criterion_scores['TA']}, CC={criterion_scores['CC']}, "
        f"LR={criterion_scores['LR']}, GRA={criterion_scores['GRA']}, "
        f"Overall={evaluation.get('overall_band', '?')}"
    )
    examiner_verdict = state.get("examiner_verdict", "")
    verdict_context = ""
    if examiner_verdict:
        verdict_context = (
            f"\nNote: The Examiner {'revised' if examiner_verdict == 'revised' else 'maintained'} "
            "the scores after pedagogical review — the final scores reflect a calibrated assessment."
        )

    prompt = HumanMessage(content=(
        f"Band scores from the Examiner: {eval_summary}{verdict_context}\n\n"
        f"The student's weakest criterion is **{weakest}** (score: {criterion_scores[weakest]}).\n\n"
        "Based on the original essay and these scores, generate a personalised lesson plan "
        "that will help this student improve their IELTS Writing. "
        "Be warm, specific, and actionable."
    ))

    tutor_feedback_dict: dict = {}
    feedback_text: str = ""

    try:
        llm_structured = llm.with_structured_output(TutorFeedback)
        feedback: TutorFeedback = llm_structured.invoke(
            [SystemMessage(content=_TUTOR_SYSTEM)] + state["messages"] + [prompt]
        )
        tutor_feedback_dict = feedback.model_dump()
        # Build a readable text version for the message history
        feedback_text = (
            f"**📚 Personalised Lesson Plan**\n\n"
            f"**Priority Focus:** {feedback.priority_focus}\n\n"
            f"**{feedback.encouragement}**\n\n"
            "**Lesson Plan:**\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(feedback.lesson_plan)) + "\n\n"
            "**For Your Next Essay:**\n" + "\n".join(f"- {t}" for t in feedback.next_essay_tips)
        )
        if feedback.rewrite_examples:
            feedback_text += "\n\n**Rewrite Examples:**\n" + "\n".join(
                f"- {r}" for r in feedback.rewrite_examples
            )
        console.print("[dim green]✅ Tutor lesson plan generated (structured)[/dim green]")
    except Exception as exc:
        console.print(f"[dim yellow]Tutor lesson plan fallback: {exc}[/dim yellow]")
        response = llm.invoke([SystemMessage(content=_TUTOR_SYSTEM)] + state["messages"] + [prompt])
        feedback_text = response.content

    return {
        "messages":      [AIMessage(content=feedback_text)],
        "tutor_feedback": tutor_feedback_dict,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Tutor router
# ══════════════════════════════════════════════════════════════════════════════

def challenge_router(state: dict) -> str:
    """After tutor_review: route to reconsider if there are challenges, else lesson_plan."""
    has_challenge = bool(state.get("tutor_challenge", ""))
    if has_challenge:
        console.print("[dim]→ Routing to examiner_reconsider[/dim]")
        return "examiner_reconsider"
    console.print("[dim]→ Routing to tutor_lesson_plan[/dim]")
    return "tutor_lesson_plan"
