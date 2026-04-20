"""
agent/schemas.py
────────────────
Pydantic models for validated, structured IELTS evaluation output.

Using these models with `llm.with_structured_output()` forces the LLM to
emit well-formed JSON instead of free text, eliminating brittle regex parsing.

Exports
───────
  RubricCriterion   — score + analysis for one IELTS criterion
  IELTSEvaluation   — full structured evaluation with all four criteria
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RubricCriterion(BaseModel):
    """Score and qualitative analysis for a single IELTS rubric criterion."""

    score: float = Field(
        ...,
        ge=0.0,
        le=9.0,
        description="IELTS band score for this criterion (0–9, 0.5 increments).",
    )
    analysis: str = Field(
        ...,
        description="Detailed examiner analysis — minimum 2 sentences.",
    )


class IELTSEvaluation(BaseModel):
    """
    Fully structured IELTS Writing Task 2 evaluation.

    All four rubric criteria are represented as validated RubricCriterion
    objects, ensuring scores stay within the 0–9 band range and analysis
    text is always present.
    """

    overall_band: float = Field(
        ...,
        ge=0.0,
        le=9.0,
        description="Overall band score (mean of the 4 criteria, rounded to nearest 0.5).",
    )
    task_achievement: RubricCriterion = Field(
        ..., description="Task Achievement (TA): addresses the prompt, position, arguments."
    )
    coherence_cohesion: RubricCriterion = Field(
        ..., description="Coherence & Cohesion (CC): logical flow, paragraphing, linking."
    )
    lexical_resource: RubricCriterion = Field(
        ..., description="Lexical Resource (LR): vocabulary range, accuracy, collocations."
    )
    grammatical_range: RubricCriterion = Field(
        ..., description="Grammatical Range & Accuracy (GRA): sentence variety, error rate."
    )
    key_strengths: list[str] = Field(
        ...,
        min_length=1,
        description="List of 2–3 concrete strengths observed in the essay.",
    )
    areas_for_improvement: list[str] = Field(
        ...,
        min_length=1,
        description="List of 2–3 prioritised improvement areas.",
    )
    rewrite_suggestions: list[str] = Field(
        default_factory=list,
        description="2–3 sentence rewrites showing BEFORE → AFTER improvements.",
    )

    # ── Convenience helpers ────────────────────────────────────────────────────

    def to_scores_dict(self) -> dict[str, float]:
        """Return a flat {key: score} dict compatible with display_score_table."""
        return {
            "Overall": self.overall_band,
            "TA":      self.task_achievement.score,
            "CC":      self.coherence_cohesion.score,
            "LR":      self.lexical_resource.score,
            "GRA":     self.grammatical_range.score,
        }

    def to_markdown(self) -> str:
        """Render the evaluation as a Markdown string for CLI display."""
        lines = [
            f"## Band Score: {self.overall_band}",
            "",
            f"### Task Achievement (TA): {self.task_achievement.score}",
            self.task_achievement.analysis,
            "",
            f"### Coherence and Cohesion (CC): {self.coherence_cohesion.score}",
            self.coherence_cohesion.analysis,
            "",
            f"### Lexical Resource (LR): {self.lexical_resource.score}",
            self.lexical_resource.analysis,
            "",
            f"### Grammatical Range and Accuracy (GRA): {self.grammatical_range.score}",
            self.grammatical_range.analysis,
            "",
            "### Key Strengths",
            *[f"- {s}" for s in self.key_strengths],
            "",
            "### Areas for Improvement",
            *[f"- {a}" for a in self.areas_for_improvement],
        ]
        if self.rewrite_suggestions:
            lines += ["", "### Rewrite Suggestions", *[f"- {r}" for r in self.rewrite_suggestions]]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Multi-agent schemas
# ══════════════════════════════════════════════════════════════════════════════

class ChallengeSignal(BaseModel):
    """
    A challenge raised by the Tutor Agent against a specific Examiner score.
    Carries concrete essay evidence so the Examiner can make an informed decision.
    """
    criterion:         str   = Field(..., description="IELTS criterion key: TA | CC | LR | GRA")
    current_score:     float = Field(..., ge=0, le=9)
    suggested_minimum: float = Field(..., ge=0, le=9, description="Tutor's suggested floor score")
    evidence:          str   = Field(..., description="Specific essay evidence supporting the challenge")


class ExaminerVerdict(BaseModel):
    """
    The Examiner's response to a Tutor challenge.
    Either revises specific scores (with justification) or holds firm (with reasoning).
    """
    maintains_scores: bool = Field(
        ..., description="True = Examiner holds original scores; False = Examiner revises"
    )
    justification: str = Field(
        ..., description="Examiner's reasoning for maintaining or changing scores"
    )
    # Optional per-criterion overrides (only populated when maintains_scores=False)
    revised_ta:      float | None = Field(None, ge=0, le=9)
    revised_cc:      float | None = Field(None, ge=0, le=9)
    revised_lr:      float | None = Field(None, ge=0, le=9)
    revised_gra:     float | None = Field(None, ge=0, le=9)
    revised_overall: float | None = Field(None, ge=0, le=9)


class TutorFeedback(BaseModel):
    """
    Student-facing pedagogical output produced by the Tutor Agent.
    Designed to be encouraging, specific, and actionable regardless of score.
    """
    priority_focus: str = Field(
        ..., description="The ONE criterion the student should prioritise: TA | CC | LR | GRA"
    )
    encouragement: str = Field(
        ..., description="Personalised 2–3 sentence motivational message referencing specifics."
    )
    lesson_plan: list[str] = Field(
        ..., min_length=3, description="3–5 specific, ordered improvement actions."
    )
    next_essay_tips: list[str] = Field(
        ..., min_length=2, description="2–3 concrete things to focus on in the NEXT essay."
    )
    rewrite_examples: list[str] = Field(
        default_factory=list,
        description="1–2 sentence rewrites: 'BEFORE: … → AFTER: …'"
    )
