"""
agent/tools.py
──────────────
LangChain @tool definitions that run automatically during essay evaluation.

Tools
─────
  count_words    — word count + minimum check (≥ 250 for Task 2)
  grammar_check  — lightweight heuristic grammar scan (run-on, missing articles)
  topic_keywords — lexical diversity + top keyword frequency

Both TOOLS (list) and TOOL_MAP (name → callable) are exported for use in
graph.py.
"""

from __future__ import annotations

import json
import re
from collections import Counter

from langchain_core.tools import tool


# ── Tool definitions ──────────────────────────────────────────────────────────

@tool
def count_words(text: str) -> str:
    """Count the number of words in an IELTS essay and warn if below 250.
    Use this tool BEFORE evaluating any essay."""
    words = text.split()
    count = len(words)
    status = (
        "✅ Meets minimum"
        if count >= 250
        else f"⚠️ Below minimum (need {250 - count} more words)"
    )
    return json.dumps({"word_count": count, "minimum_required": 250, "status": status})


@tool
def grammar_check(text: str) -> str:
    """Scan an essay for common grammatical issues (subject-verb agreement,
    article misuse, run-on sentences). Returns a JSON summary of findings.
    Use this for a preliminary grammar scan before scoring GRA."""
    issues: list[dict] = []
    sentences = re.split(r"(?<=[.!?])\s+", text)

    for i, sent in enumerate(sentences, 1):
        words = sent.split()
        # Very long sentence → possible run-on
        if len(words) > 50:
            issues.append({
                "sentence": i,
                "issue": "Possible run-on sentence",
                "snippet": sent[:60] + "…",
            })
        # Missing article before adjective (rough heuristic)
        if re.search(r"\b(is|are|was|were)\s+(important|effective|essential|necessary)\b", sent, re.I):
            if not re.search(
                r"\b(an?|the|this|that|their|its)\s+(important|effective|essential|necessary)\b",
                sent, re.I,
            ):
                issues.append({
                    "sentence": i,
                    "issue": "Possible missing article",
                    "snippet": sent[:60] + "…",
                })

    summary = {
        "total_sentences": len(sentences),
        "issues_found": len(issues),
        "details": issues[:5],  # cap at 5 to keep context small
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


@tool
def topic_keywords(text: str) -> str:
    """Extract the top topic-related keywords from an essay for Lexical Resource
    analysis. Returns word frequency data (excluding common stopwords).
    Use this before scoring LR."""
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "that", "this", "these", "those", "it", "its", "they",
        "their", "them", "we", "our", "us", "i", "my", "me", "you", "your",
        "he", "she", "his", "her", "not", "no", "so", "as", "if", "than",
        "more", "also", "very", "can", "many", "some", "such",
    }
    tokens = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    filtered = [t for t in tokens if t not in STOPWORDS]
    freq = Counter(filtered).most_common(15)
    unique_ratio = round(len(set(filtered)) / max(len(filtered), 1), 2)
    return json.dumps(
        {
            "top_keywords": dict(freq),
            "unique_word_ratio": unique_ratio,
            "lexical_diversity_note": (
                "Good lexical diversity"
                if unique_ratio > 0.6
                else "Consider using more varied vocabulary"
            ),
        },
        ensure_ascii=False,
        indent=2,
    )


# ── Registry ──────────────────────────────────────────────────────────────────

TOOLS = [count_words, grammar_check, topic_keywords]
TOOL_MAP: dict[str, object] = {t.name: t for t in TOOLS}
