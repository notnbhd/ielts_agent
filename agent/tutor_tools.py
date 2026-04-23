"""
Tutor-specific LangChain tools used by the Tutor ReAct agent.

These are mock utilities so local/offline development can exercise
planning + tool-calling without external services.
"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def search_student_history(student_id: str, error_type: str) -> str:
    """Return mock historical mistakes for a student and error type."""
    sid = (student_id or "anonymous").strip().lower()
    etype = (error_type or "general").strip().lower()

    student_profiles = {
        "anonymous": {
            "grammar": "Frequent article/preposition slips in introductions.",
            "vocabulary": "Repeats high-frequency words and avoids collocations.",
            "coherence": "Body paragraphs often mix two ideas without clear topic sentences.",
            "task_response": "Examples are sometimes generic and not tightly linked to the prompt.",
            "general": "Inconsistent proofreading and limited post-writing self-check.",
        },
        "student-001": {
            "grammar": "Improved tense control recently, but still drops articles before singular nouns.",
            "vocabulary": "Strong topic words in technology essays; weak range in education topics.",
            "coherence": "Uses linking words, but paragraph progression can feel abrupt.",
            "task_response": "Maintains clear opinion but support details are occasionally underdeveloped.",
            "general": "Shows progress when using planning templates before drafting.",
        },
    }

    profile = student_profiles.get(sid, student_profiles["anonymous"])
    note = profile.get(etype, profile["general"])

    return (
        f"Student={sid}; error_type={etype}. "
        f"Historical pattern: {note} "
        "Recommended coaching angle: target one repeated error family per essay."
    )


@tool
def generate_targeted_exercise(topic: str, difficulty: str) -> str:
    """Return a short mock exercise tailored to a topic and difficulty."""
    t = (topic or "writing_fundamentals").strip().lower()
    d = (difficulty or "intermediate").strip().lower()

    exercise_bank = {
        "grammar": {
            "basic": (
                "Fill in articles (a/an/the):\n"
                "1) ___ university should teach practical skills.\n"
                "2) Technology is ___ useful tool for learners.\n"
                "Answer key: 1) A 2) a"
            ),
            "intermediate": (
                "Sentence correction:\n"
                "BEFORE: People should to spend less time in social media.\n"
                "Task: Rewrite with correct grammar and natural form.\n"
                "Sample: People should spend less time on social media."
            ),
            "advanced": (
                "Complex sentence drill:\n"
                "Combine ideas with accurate clause control and punctuation:\n"
                "Idea A: Governments should fund public transport.\n"
                "Idea B: This reduces traffic and emissions."
            ),
        },
        "vocabulary": {
            "basic": (
                "Synonym swap mini-task:\n"
                "Replace 'good/bad/important' with precise alternatives in 3 sentences."
            ),
            "intermediate": (
                "Collocation drill:\n"
                "Complete phrases: 'address ___ issue', 'play a ___ role', 'reach a ___'."
            ),
            "advanced": (
                "Paraphrase challenge:\n"
                "Rewrite one thesis statement using advanced but natural lexical variation."
            ),
        },
        "coherence": {
            "basic": "Write one clear topic sentence and one supporting sentence for a body paragraph.",
            "intermediate": "Reorder 5 jumbled sentences into a logical paragraph with cohesive flow.",
            "advanced": "Create a two-paragraph argument with explicit progression and contrast.",
        },
        "task_response": {
            "basic": "List 2 direct reasons that answer the prompt, then add one specific example each.",
            "intermediate": "Write a thesis + two argument points, each tightly linked to the question wording.",
            "advanced": "Draft a balanced argument and justify your final position with nuanced evidence.",
        },
        "writing_fundamentals": {
            "basic": "Write 4 sentences: opinion, reason, example, mini-conclusion.",
            "intermediate": "Plan a 4-paragraph essay outline in 4 minutes.",
            "advanced": "Write one concise introduction with a clear thesis and scope.",
        },
    }

    by_topic = exercise_bank.get(t, exercise_bank["writing_fundamentals"])
    body = by_topic.get(d, by_topic["intermediate"])

    return f"Exercise topic={t}, difficulty={d}.\n{body}"


TUTOR_TOOLS = [search_student_history, generate_targeted_exercise]
TUTOR_TOOL_MAP: dict[str, object] = {tool_.name: tool_ for tool_ in TUTOR_TOOLS}
