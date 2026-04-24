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

import difflib
import json
import re
from collections import Counter
from typing import Any

from langchain_core.tools import tool


_GEC_MODEL_ID = "vennify/t5-base-grammar-correction"
_GEC_RUNTIME: dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "ready": False,
    "error": "",
}


def _load_gec_runtime() -> dict[str, Any]:
    """Load GEC model lazily so normal app startup stays fast."""
    if _GEC_RUNTIME["ready"] or _GEC_RUNTIME["error"]:
        return _GEC_RUNTIME

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(_GEC_MODEL_ID)
        model = AutoModelForSeq2SeqLM.from_pretrained(_GEC_MODEL_ID)
        _GEC_RUNTIME["tokenizer"] = tokenizer
        _GEC_RUNTIME["model"] = model
        _GEC_RUNTIME["ready"] = True
    except Exception as exc:
        _GEC_RUNTIME["error"] = str(exc)

    return _GEC_RUNTIME


def _gec_correct_sentence(sentence: str) -> str:
    """Return corrected sentence from a seq2seq GEC model, or input on failure."""
    runtime = _load_gec_runtime()
    if not runtime["ready"]:
        return sentence

    tokenizer = runtime["tokenizer"]
    model = runtime["model"]

    prompt = f"grammar: {sentence.strip()}"

    try:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        output_ids = model.generate(
            **encoded,
            max_new_tokens=128,
            num_beams=4,
            early_stopping=True,
        )
        corrected = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return corrected or sentence
    except Exception:
        return sentence


def _estimate_edit_count(original: str, corrected: str) -> int:
    """Approximate number of token-level edits between two sentence variants."""
    a = original.split()
    b = corrected.split()
    sm = difflib.SequenceMatcher(a=a, b=b)
    edits = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        edits += max(i2 - i1, j2 - j1)
    return edits


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
    """Run grammar diagnostics using a GEC model plus lightweight heuristics.

    Returns a JSON summary with detected issues. If the GEC model is not
    available, falls back to heuristic checks only.
    """
    issues: list[dict] = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    runtime = _load_gec_runtime()
    using_gec = bool(runtime.get("ready"))

    for i, sent in enumerate(sentences, 1):
        if not sent.strip():
            continue

        words = sent.split()
        # Very long sentence: possible run-on.
        if len(words) > 50:
            issues.append({
                "sentence": i,
                "issue": "Possible run-on sentence",
                "snippet": sent[:60] + "...",
            })

        # GEC model-driven correction signal.
        corrected = _gec_correct_sentence(sent)
        if corrected.strip() and corrected.strip() != sent.strip():
            edits = _estimate_edit_count(sent, corrected)
            issues.append({
                "sentence": i,
                "issue": "GEC correction suggested",
                "snippet": sent[:60] + "...",
                "suggestion": corrected[:160],
                "edit_count": edits,
            })

        # Keep a cheap heuristic fallback signal for article issues.
        if re.search(r"\b(is|are|was|were)\s+(important|effective|essential|necessary)\b", sent, re.I):
            if not re.search(
                r"\b(an?|the|this|that|their|its)\s+(important|effective|essential|necessary)\b",
                sent, re.I,
            ):
                issues.append({
                    "sentence": i,
                    "issue": "Possible missing article",
                    "snippet": sent[:60] + "...",
                })

    summary = {
        "total_sentences": len([s for s in sentences if s.strip()]),
        "issues_found": len(issues),
        "details": issues[:5],  # cap at 5 to keep context small
        "gec_model": _GEC_MODEL_ID,
        "gec_enabled": using_gec,
        "gec_error": "" if using_gec else runtime.get("error", ""),
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
