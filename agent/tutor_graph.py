"""
Tutor Agent nodes for the multi-agent supervisor.

Tutor responsibilities:
  1. Challenge Review (deterministic rules, no LLM)
  2. Lesson Planning via ReAct (autonomous planning + tool calling)
  3. Structured Exit (final output always validated as TutorFeedback)

Exported nodes (called by supervisor.py)
─────────────────────────────────────────
  tutor_review_node        — produces tutor_challenge (list serialised to str)
  examiner_reconsider_node — examiner considers challenge, returns ExaminerVerdict
  tutor_lesson_plan_node   — runs Tutor ReAct + final TutorFeedback formatter

Exported routers
────────────────
  challenge_router         — routes to reconsider or lesson_plan after review
"""

from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from rich.console import Console

from agent.schemas import ChallengeSignal, ExaminerVerdict, TutorFeedback
from agent.tutor_tools import TUTOR_TOOLS, generate_targeted_exercise, search_student_history

console = Console()
_MAX_REACT_LOOPS = 5


# ══════════════════════════════════════════════════════════════════════════════
# Tutor system prompts
# ══════════════════════════════════════════════════════════════════════════════

_TUTOR_REACT_SYSTEM = """\
You are an autonomous IELTS Writing Tutor agent using a ReAct workflow.

Always execute this process:
1) Identify the weakest criterion from the final IELTS evaluation.
2) Call tools to gather extra evidence (student history and targeted practice).
3) Synthesize findings into concise planning notes for a formatter model.

Rules:
- Use tools before giving your final internal summary.
- Be specific, practical, and growth-focused.
- Keep reasoning short and avoid unnecessary repetition.
- Do NOT produce final JSON for the student in this step.
"""

_TUTOR_FORMATTER_SYSTEM = """\
You are a strict JSON formatter for IELTS tutoring output.
You must return valid data matching the TutorFeedback schema exactly:
- priority_focus: one of TA | CC | LR | GRA
- encouragement: 2-3 personalised sentences
- lesson_plan: 3-5 ordered action items
- next_essay_tips: 2-3 concrete tips
- rewrite_examples: optional 0-2 rewrite examples

Ground your plan in the evaluation and ReAct evidence provided.
"""

_EXAMINER_RECONSIDER_SYSTEM = """\
You are an impartial IELTS Writing examiner reviewing a pedagogical challenge
raised by the student's tutor. The tutor has identified essay evidence that may
warrant a score adjustment.

Your task: weigh the evidence objectively against the official IELTS descriptors.
- If the evidence is valid and the score was too strict -> revise it (be fair).
- If the score is correct despite the evidence -> maintain it with a clear justification.

Do NOT inflate scores to be kind. Accuracy is paramount.
"""


# ══════════════════════════════════════════════════════════════════════════════
# Rule-based challenge detection (no LLM call - fast + reliable)
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


def _clip(text: str, limit: int = 500) -> str:
    """Keep debug text compact so message history and checkpoints stay small."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _extract_essay_text(state: dict) -> str:
    """Extract essay text from the most recent human message payload."""
    essay_msg = next(
        (m for m in state.get("messages", []) if isinstance(m, HumanMessage) and "**Essay:**" in m.content),
        None,
    )
    if essay_msg is None:
        return ""
    match = re.search(r"\*\*Essay:\*\*\s*(.*)", essay_msg.content, re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def _criterion_scores(evaluation: dict) -> dict[str, float]:
    """Return TA/CC/LR/GRA scores with safe defaults."""
    return {
        "TA": float(evaluation.get("task_achievement", {}).get("score", 0.0)),
        "CC": float(evaluation.get("coherence_cohesion", {}).get("score", 0.0)),
        "LR": float(evaluation.get("lexical_resource", {}).get("score", 0.0)),
        "GRA": float(evaluation.get("grammatical_range", {}).get("score", 0.0)),
    }


def _build_react_brief(state: dict, evaluation: dict, weakest: str) -> str:
    """Build a compact task brief for the Tutor ReAct sub-agent."""
    scores = _criterion_scores(evaluation)
    student_id = str(state.get("student_id", "anonymous"))
    examiner_verdict = state.get("examiner_verdict", "")
    essay_text = _extract_essay_text(state)
    essay_excerpt = _clip(essay_text, 1200)

    verdict_note = ""
    if examiner_verdict:
        verdict_note = (
            f"Examiner verdict after Tutor challenge: "
            f"{'revised' if examiner_verdict == 'revised' else 'held'} scores."
        )

    return (
        f"Student ID: {student_id}\n"
        f"Final scores: TA={scores['TA']}, CC={scores['CC']}, LR={scores['LR']}, "
        f"GRA={scores['GRA']}, Overall={evaluation.get('overall_band', '?')}\n"
        f"Weakest criterion: {weakest}\n"
        f"{verdict_note}\n\n"
        "Task:\n"
        "- Analyze why the weakest criterion is holding this student back.\n"
        "- Call tools to retrieve student history and produce one focused exercise.\n"
        "- Return concise planning notes for the formatter model.\n\n"
        "Essay excerpt:\n"
        f"{essay_excerpt}"
    )


def _extract_react_steps(
    messages: list[Any],
    *,
    source: str = "react",
    start_index: int = 1,
) -> tuple[list[dict[str, Any]], bool]:
    """Capture a concise ReAct trace from agent messages for debugging."""
    steps: list[dict[str, Any]] = []
    used_tools = False
    step_index = start_index

    for msg in messages:
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", []) or []
            if tool_calls:
                for call in tool_calls:
                    used_tools = True
                    steps.append({
                        "step_index": step_index,
                        "source": source,
                        "step_type": "tool_call",
                        "tool": call.get("name", "unknown"),
                        "args": call.get("args", {}),
                        "duration_ms": 0.0,
                    })
                    step_index += 1

            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content and content.strip():
                steps.append({
                    "step_index": step_index,
                    "source": source,
                    "step_type": "thought",
                    "content": _clip(content, 320),
                    "duration_ms": 0.0,
                })
                step_index += 1

        elif isinstance(msg, ToolMessage):
            used_tools = True
            tool_name = getattr(msg, "name", "") or msg.additional_kwargs.get("name", "tool")
            steps.append({
                "step_index": step_index,
                "source": source,
                "step_type": "tool_result",
                "tool": tool_name,
                "content": _clip(str(msg.content), 600),
                "duration_ms": 0.0,
            })
            step_index += 1

    return steps, used_tools


def _summarise_react_messages(messages: list[Any]) -> str:
    """Aggregate useful tool findings and final agent notes into compact text."""
    chunks: list[str] = []
    final_notes = ""

    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "") or msg.additional_kwargs.get("name", "tool")
            chunks.append(f"{tool_name}: {_clip(str(msg.content), 700)}")
        elif isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content and content.strip():
                final_notes = _clip(content, 1200)

    if final_notes:
        chunks.append(f"Tutor agent notes: {final_notes}")

    return "\n\n".join(chunks).strip()


def _fallback_tool_sequence(
    student_id: str,
    weakest: str,
    weakest_score: float,
    *,
    start_index: int = 1,
) -> tuple[list[dict[str, Any]], str]:
    """Deterministic fallback when local models fail to emit stable tool calls."""
    topic_map = {
        "TA": ("task_response", "task_response"),
        "CC": ("coherence", "coherence"),
        "LR": ("vocabulary", "vocabulary"),
        "GRA": ("grammar", "grammar"),
    }
    topic, error_type = topic_map.get(weakest, ("writing_fundamentals", "general"))

    if weakest_score < 5.5:
        difficulty = "basic"
    elif weakest_score < 6.5:
        difficulty = "intermediate"
    else:
        difficulty = "advanced"

    steps: list[dict[str, Any]] = []
    step_index = start_index

    t0 = perf_counter()
    try:
        history = search_student_history.invoke({
            "student_id": student_id,
            "error_type": error_type,
        })
        elapsed_ms = round((perf_counter() - t0) * 1000, 2)
        steps.append({
            "step_index": step_index,
            "source": "fallback",
            "step_type": "tool_call",
            "tool": "search_student_history",
            "args": {"student_id": student_id, "error_type": error_type},
            "duration_ms": 0.0,
        })
        step_index += 1
        steps.append({
            "step_index": step_index,
            "source": "fallback",
            "step_type": "tool_result",
            "tool": "search_student_history",
            "content": _clip(str(history), 700),
            "duration_ms": elapsed_ms,
        })
        step_index += 1
    except Exception as exc:
        elapsed_ms = round((perf_counter() - t0) * 1000, 2)
        history = f"search_student_history failed: {exc}"
        steps.append({
            "step_index": step_index,
            "source": "fallback",
            "step_type": "tool_error",
            "tool": "search_student_history",
            "content": _clip(history, 300),
            "duration_ms": elapsed_ms,
        })
        step_index += 1

    t0 = perf_counter()
    try:
        exercise = generate_targeted_exercise.invoke({
            "topic": topic,
            "difficulty": difficulty,
        })
        elapsed_ms = round((perf_counter() - t0) * 1000, 2)
        steps.append({
            "step_index": step_index,
            "source": "fallback",
            "step_type": "tool_call",
            "tool": "generate_targeted_exercise",
            "args": {"topic": topic, "difficulty": difficulty},
            "duration_ms": 0.0,
        })
        step_index += 1
        steps.append({
            "step_index": step_index,
            "source": "fallback",
            "step_type": "tool_result",
            "tool": "generate_targeted_exercise",
            "content": _clip(str(exercise), 700),
            "duration_ms": elapsed_ms,
        })
        step_index += 1
    except Exception as exc:
        elapsed_ms = round((perf_counter() - t0) * 1000, 2)
        exercise = f"generate_targeted_exercise failed: {exc}"
        steps.append({
            "step_index": step_index,
            "source": "fallback",
            "step_type": "tool_error",
            "tool": "generate_targeted_exercise",
            "content": _clip(exercise, 300),
            "duration_ms": elapsed_ms,
        })
        step_index += 1

    summary = (
        "Fallback tool execution was used due unstable/no tool-calling behavior.\n"
        f"search_student_history => {history}\n"
        f"generate_targeted_exercise => {exercise}"
    )
    return steps, summary


def _fallback_tutor_feedback(weakest: str, react_summary: str) -> TutorFeedback:
    """Guarantee schema-compliant TutorFeedback even if structured call fails."""
    encouragement = (
        "You are closer than you think. Your score profile shows a clear target, "
        "and focused practice on one criterion can quickly lift the overall band."
    )
    if react_summary:
        encouragement += " I used your recent error patterns to tailor the plan below."

    return TutorFeedback(
        priority_focus=weakest,
        encouragement=encouragement,
        lesson_plan=[
            f"Review your last two essays and label every {weakest} weakness in the margin.",
            "Complete one short targeted exercise and check every answer with explanation.",
            "Rewrite one body paragraph with clearer control of your priority criterion.",
        ],
        next_essay_tips=[
            f"Before writing, set one measurable goal for {weakest} and monitor it while drafting.",
            "After writing, run a 5-minute self-check focused only on your weakest criterion.",
        ],
        rewrite_examples=[],
    )


def _render_tutor_feedback(feedback: TutorFeedback) -> str:
    """Render structured TutorFeedback into readable markdown for chat history."""
    lines = [
        "**Personalised Lesson Plan**",
        "",
        f"**Priority Focus:** {feedback.priority_focus}",
        "",
        f"**{feedback.encouragement}**",
        "",
        "**Lesson Plan:**",
    ]
    lines += [f"{i + 1}. {step}" for i, step in enumerate(feedback.lesson_plan)]
    lines += ["", "**For Your Next Essay:**"]
    lines += [f"- {tip}" for tip in feedback.next_essay_tips]

    if feedback.rewrite_examples:
        lines += ["", "**Rewrite Examples:**"]
        lines += [f"- {example}" for example in feedback.rewrite_examples]

    return "\n".join(lines)


def _detect_challenges(essay: str, evaluation: dict) -> list[ChallengeSignal]:
    """
    Produce a list of ChallengeSignals where essay evidence contradicts
    a seemingly low Examiner score. All checks are rule-based.
    """
    challenges: list[ChallengeSignal] = []
    essay_lower = essay.lower()

    ta_score = evaluation.get("task_achievement", {}).get("score", 0.0)
    cc_score = evaluation.get("coherence_cohesion", {}).get("score", 0.0)
    lr_score = evaluation.get("lexical_resource", {}).get("score", 0.0)

    has_position = any(re.search(p, essay_lower) for p in _THESIS_PATTERNS)
    paragraphs = [p for p in re.split(r"\n\s*\n", essay.strip()) if p.strip()]
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
    TUTOR REVIEW node - checks Examiner scores against essay evidence.
    Produces a serialised list of ChallengeSignal dicts stored in
    state["tutor_challenge"]. Empty string = no challenge.
    """
    evaluation = state.get("evaluation", {})
    essay_text = _extract_essay_text(state)

    if not evaluation or not essay_text:
        console.print("[dim]Tutor review: no evaluation or essay found, skipping challenges[/dim]")
        return {"tutor_challenge": ""}

    challenges = _detect_challenges(essay_text, evaluation)

    if challenges:
        payload = json.dumps([c.model_dump() for c in challenges], ensure_ascii=False)
        console.print(
            f"[yellow]Tutor raises {len(challenges)} challenge(s): "
            f"{', '.join(c.criterion for c in challenges)}[/yellow]"
        )
        return {"tutor_challenge": payload}

    console.print("[dim green]Tutor review: no challenges, scores look fair[/dim green]")
    return {"tutor_challenge": ""}


def examiner_reconsider_node(state: dict, *, model: str, temperature: float) -> dict:
    """
    EXAMINER RECONSIDER node - the Examiner reads the Tutor's challenge and
    produces an ExaminerVerdict (structured output): either revises scores
    or holds firm with a clear rubric-based justification.

    If the Examiner revises, the evaluation dict is updated in state so the
    Tutor's lesson plan reflects the final agreed scores.
    """
    llm = ChatOllama(model=model, temperature=0.1, num_predict=1024, num_ctx=4096)
    evaluation = state.get("evaluation", {})
    challenges_raw = state.get("tutor_challenge", "")

    try:
        challenges = [ChallengeSignal(**c) for c in json.loads(challenges_raw)]
    except Exception:
        return {"tutor_challenge": ""}

    challenge_text = "\n".join(
        f"- [{c.criterion}] Current: {c.current_score} | Suggested floor: {c.suggested_minimum}\n"
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
        console.print("[cyan]Examiner revised scores after Tutor challenge[/cyan]")
        field_map = {
            "revised_ta": ("task_achievement", "score"),
            "revised_cc": ("coherence_cohesion", "score"),
            "revised_lr": ("lexical_resource", "score"),
            "revised_gra": ("grammatical_range", "score"),
        }
        for attr, (criterion_key, score_key) in field_map.items():
            new_val = getattr(verdict, attr, None)
            if new_val is not None and criterion_key in updated_eval:
                updated_eval[criterion_key] = dict(updated_eval[criterion_key])
                updated_eval[criterion_key][score_key] = new_val
        if verdict.revised_overall is not None:
            updated_eval["overall_band"] = verdict.revised_overall
    else:
        console.print("[dim]Examiner held original scores[/dim]")

    verdict_note = AIMessage(content=(
        "**Examiner response to Tutor challenge:**\n"
        f"{'Scores revised' if not verdict.maintains_scores else 'Scores maintained'}\n"
        f"_{verdict.justification}_"
    ))
    return {
        "messages": [verdict_note],
        "evaluation": updated_eval,
        "tutor_challenge": "",
        "examiner_verdict": "revised" if not verdict.maintains_scores else "held",
    }


def tutor_lesson_plan_node(
    state: dict,
    *,
    model: str,
    temperature: float,
    max_react_loops: int = _MAX_REACT_LOOPS,
) -> dict:
    """
    TUTOR LESSON PLAN node (hybrid design).

    Phase A: Run Tutor ReAct sub-graph (create_react_agent) with tool calling.
    Phase B: Force structured exit using TutorFeedback schema.

    If local model tool-calling is unstable, a deterministic fallback tool
    sequence runs so the tutor still gets external context before formatting.
    """
    evaluation = state.get("evaluation", {})
    scores = _criterion_scores(evaluation)
    weakest = min(scores, key=scores.get)
    total_start = perf_counter()

    react_steps: list[dict[str, Any]] = []
    react_summary = ""
    used_tools = False
    used_fallback = False
    react_duration_ms = 0.0
    formatter_duration_ms = 0.0
    react_error = ""
    student_id = str(state.get("student_id", "anonymous"))

    llm_react = ChatOllama(model=model, temperature=temperature, num_predict=1024, num_ctx=4096)
    react_brief = _build_react_brief(state, evaluation, weakest)

    try:
        react_start = perf_counter()
        react_agent = create_react_agent(llm_react, TUTOR_TOOLS, prompt=_TUTOR_REACT_SYSTEM)
        react_result = react_agent.invoke(
            {"messages": [HumanMessage(content=react_brief)]},
            config={"recursion_limit": max_react_loops},
        )
        react_duration_ms = round((perf_counter() - react_start) * 1000, 2)
        react_messages = react_result.get("messages", [])
        react_steps, used_tools = _extract_react_steps(
            react_messages,
            source="react",
            start_index=1,
        )
        react_summary = _summarise_react_messages(react_messages)
        console.print(
            f"[dim green]Tutor ReAct finished with {len(react_steps)} trace step(s) "
            f"(limit={max_react_loops})[/dim green]"
        )
    except Exception as exc:
        react_error = str(exc)
        console.print(f"[dim yellow]Tutor ReAct fallback (tool-calling issue): {exc}[/dim yellow]")

    if not used_tools:
        used_fallback = True
        fallback_steps, fallback_summary = _fallback_tool_sequence(
            student_id=student_id,
            weakest=weakest,
            weakest_score=scores.get(weakest, 0.0),
            start_index=len(react_steps) + 1,
        )
        react_steps.extend(fallback_steps)
        react_summary = (react_summary + "\n\n" + fallback_summary).strip()

    formatter_prompt = HumanMessage(content=(
        "Build the final tutoring output from the data below.\n\n"
        f"Final IELTS evaluation: {json.dumps(evaluation, ensure_ascii=False)}\n"
        f"Weakest criterion: {weakest}\n"
        f"Tutor ReAct trace: {json.dumps(react_steps, ensure_ascii=False)}\n\n"
        "Synthesised ReAct evidence:\n"
        f"{react_summary}\n\n"
        "Return TutorFeedback only."
    ))

    llm_formatter = ChatOllama(
        model=model,
        temperature=min(max(temperature, 0.1), 0.4),
        num_predict=1536,
        num_ctx=4096,
    )

    feedback_obj: TutorFeedback
    try:
        formatter_start = perf_counter()
        llm_structured = llm_formatter.with_structured_output(TutorFeedback)
        raw_feedback = llm_structured.invoke([
            SystemMessage(content=_TUTOR_FORMATTER_SYSTEM),
            formatter_prompt,
        ])
        feedback_obj = TutorFeedback.model_validate(raw_feedback)
        formatter_duration_ms = round((perf_counter() - formatter_start) * 1000, 2)
        console.print("[dim green]Tutor structured formatter succeeded[/dim green]")
    except Exception as exc:
        formatter_duration_ms = round((perf_counter() - formatter_start) * 1000, 2)
        console.print(f"[dim yellow]Tutor structured formatter fallback: {exc}[/dim yellow]")
        feedback_obj = _fallback_tutor_feedback(weakest, react_summary)

    total_duration_ms = round((perf_counter() - total_start) * 1000, 2)
    react_meta: dict[str, Any] = {
        "loop_limit": max_react_loops,
        "steps_count": len(react_steps),
        "react_duration_ms": react_duration_ms,
        "formatter_duration_ms": formatter_duration_ms,
        "total_duration_ms": total_duration_ms,
        "used_tools_in_react": used_tools,
        "used_fallback": used_fallback,
        "fallback_reason": "no_tool_usage_or_error" if used_fallback else "",
    }
    if react_error:
        react_meta["react_error"] = _clip(react_error, 240)

    return {
        "messages": [AIMessage(content=_render_tutor_feedback(feedback_obj))],
        "tutor_feedback": feedback_obj.model_dump(),
        "tutor_react_steps": react_steps,
        "tutor_react_meta": react_meta,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Tutor router
# ══════════════════════════════════════════════════════════════════════════════

def challenge_router(state: dict) -> str:
    """After tutor_review: route to reconsider if there are challenges, else lesson_plan."""
    has_challenge = bool(state.get("tutor_challenge", ""))
    if has_challenge:
        console.print("[dim]Routing to examiner_reconsider[/dim]")
        return "examiner_reconsider"
    console.print("[dim]Routing to tutor_lesson_plan[/dim]")
    return "tutor_lesson_plan"
