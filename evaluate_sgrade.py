"""
S-Grade Evaluation Script for IELTS Writing Task 2 Dataset

Reads essays from the test CSV, sends each to the IELTS tutor model via Ollama,
parses the predicted band score, and writes results to a CSV in the expected format:

    essay_id,predicted_score
    D_Ielts_Writing_Task_2_Dataset_test_0,3.5
    D_Ielts_Writing_Task_2_Dataset_test_1,4.2
    ...

Usage:
    python evaluate_sgrade.py                          # default settings
    python evaluate_sgrade.py --model qwen3.5:9B       # custom model
    python evaluate_sgrade.py --output results.csv     # custom output path
    python evaluate_sgrade.py --resume                 # resume from last checkpoint
    python evaluate_sgrade.py --limit 10               # only evaluate first 10 essays
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
)

console = Console()

# ── Constants ────────────────────────────────────────────────────────────────

DATASET_PREFIX = "D_Ielts_Writing_Task_2_Dataset_test"
OLLAMA_URL = "http://127.0.0.1:11434"

# Concise system prompt to save context tokens
EVAL_SYSTEM_PROMPT = """You are an expert IELTS Writing examiner.
Evaluate the essay on TA, CC, LR, GRA criteria (band 1.0-9.0, 0.5 steps).

ALWAYS start your response EXACTLY like this:
## Band Score: X.X
### Task Achievement (TA): X.X
### Coherence and Cohesion (CC): X.X
### Lexical Resource (LR): X.X
### Grammatical Range and Accuracy (GRA): X.X

Then give brief feedback."""

CHECKPOINT_FILE = "sgrade_checkpoint.json"

# ── Ollama Health ────────────────────────────────────────────────────────────

def is_ollama_alive() -> bool:
    """Check if Ollama server is responding."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def ensure_ollama_running(max_wait: int = 30) -> bool:
    """Start Ollama if not running and wait for it to be ready."""
    if is_ollama_alive():
        return True

    console.print("[yellow]  🔄 Ollama not responding. Restarting...[/yellow]")
    # Kill any zombie processes
    subprocess.run(["pkill", "-f", "ollama"], capture_output=True)
    time.sleep(2)

    # Start fresh
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=open("/tmp/ollama_serve.log", "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    # Wait for ready
    for i in range(max_wait):
        time.sleep(1)
        if is_ollama_alive():
            console.print("[green]  ✓ Ollama restarted successfully[/green]")
            return True

    console.print("[red]  ✗ Ollama failed to start after {max_wait}s[/red]")
    return False


# ── Score Parsing ────────────────────────────────────────────────────────────

def parse_overall_score(response: str) -> float | None:
    """Extract the overall band score from model response."""
    patterns = [
        r"Band Score:\s*([0-9]+(?:\.[0-9]+)?)",
        r"Overall\s*(?:Band)?\s*(?:Score)?:\s*([0-9]+(?:\.[0-9]+)?)",
        r"overall\s*score\s*(?:is|:)\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                if 1.0 <= score <= 9.0:
                    return score
            except ValueError:
                continue

    # Fallback: compute from sub-scores
    sub_patterns = {
        "TA":  r"Task Achievement.*?:\s*([0-9]+(?:\.[0-9]+)?)",
        "CC":  r"Coherence and Cohesion.*?:\s*([0-9]+(?:\.[0-9]+)?)",
        "LR":  r"Lexical Resource.*?:\s*([0-9]+(?:\.[0-9]+)?)",
        "GRA": r"Grammatical Range.*?:\s*([0-9]+(?:\.[0-9]+)?)",
    }
    sub_scores = []
    for pat in sub_patterns.values():
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            try:
                s = float(m.group(1))
                if 1.0 <= s <= 9.0:
                    sub_scores.append(s)
            except ValueError:
                pass

    if len(sub_scores) >= 3:
        avg = sum(sub_scores) / len(sub_scores)
        return round(avg * 2) / 2  # round to nearest 0.5

    return None


def parse_all_scores(response: str) -> dict:
    """Extract all scores for logging."""
    scores = {}
    patterns = {
        "Overall": r"Band Score:\s*([0-9]+(?:\.[0-9]+)?)",
        "TA":      r"Task Achievement.*?:\s*([0-9]+(?:\.[0-9]+)?)",
        "CC":      r"Coherence and Cohesion.*?:\s*([0-9]+(?:\.[0-9]+)?)",
        "LR":      r"Lexical Resource.*?:\s*([0-9]+(?:\.[0-9]+)?)",
        "GRA":     r"Grammatical Range.*?:\s*([0-9]+(?:\.[0-9]+)?)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                scores[key] = float(match.group(1))
            except ValueError:
                pass
    return scores


# ── Checkpoint Management ────────────────────────────────────────────────────

def load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"results": {}, "errors": [], "last_index": -1}


def save_checkpoint(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Single Essay Evaluation ─────────────────────────────────────────────────

def evaluate_essay(
    llm: ChatOllama,
    prompt: str,
    essay: str,
    max_retries: int = 3,
    retry_delay: float = 10.0,
) -> tuple[float | None, str]:
    """Evaluate one essay. Returns (score, raw_response)."""
    # Truncate very long essays to fit in context window
    max_essay_chars = 3000
    if len(essay) > max_essay_chars:
        essay = essay[:max_essay_chars] + "\n[...truncated]"

    user_input = (
        f"Evaluate this IELTS Writing Task 2 essay.\n\n"
        f"Task: {prompt}\n\n"
        f"Essay:\n{essay}"
    )

    messages = [
        SystemMessage(content=EVAL_SYSTEM_PROMPT),
        HumanMessage(content=user_input),
    ]

    last_error = None
    raw = ""

    for attempt in range(max_retries):
        # Make sure Ollama is alive before every attempt
        if not ensure_ollama_running():
            last_error = "Ollama failed to start"
            time.sleep(retry_delay)
            continue

        try:
            response = llm.invoke(messages)
            raw = response.content
            score = parse_overall_score(raw)

            if score is not None:
                return score, raw

            # Score not found in response — try a quick follow-up
            retry_msgs = messages + [
                HumanMessage(content="State only the overall band score as a number:")
            ]
            response2 = llm.invoke(retry_msgs)
            raw2 = response2.content
            # Try to grab a bare number
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)", raw2)
            if m:
                s = float(m.group(1))
                if 1.0 <= s <= 9.0:
                    return s, raw + "\n[FOLLOWUP] " + raw2

            return None, raw

        except Exception as e:
            last_error = str(e)
            console.print(
                f"[yellow]  ⚠ Attempt {attempt + 1}/{max_retries} failed: {e}[/yellow]"
            )
            # Ollama likely crashed — wait and restart
            time.sleep(retry_delay)

    return None, f"ALL_RETRIES_FAILED: {last_error}"


# ── Data I/O ─────────────────────────────────────────────────────────────────

def load_test_data(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def write_output_csv(output_path: str, results: dict, total_essays: int):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["essay_id", "predicted_score"])
        for i in range(total_essays):
            essay_id = f"{DATASET_PREFIX}_{i}"
            score = results.get(str(i), 5.0)
            writer.writerow([essay_id, score])


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="S-Grade Evaluation for IELTS Writing Task 2"
    )
    parser.add_argument("--model", default="qwen3.5:9B")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--input", default="writting_task2_dataset/test.csv")
    parser.add_argument("--output", default="sgrade_predictions.csv")
    parser.add_argument("--checkpoint", default=CHECKPOINT_FILE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-ctx", type=int, default=2048, help="Context window")
    parser.add_argument("--num-predict", type=int, default=1024, help="Max generation")
    args = parser.parse_args()

    # ── Load data ──
    console.print(f"\n[cyan]📄 Loading test data from: [bold]{args.input}[/bold][/cyan]")
    essays = load_test_data(args.input)
    total = min(len(essays), args.limit) if args.limit else len(essays)
    console.print(f"[cyan]   Found {len(essays)} essays, evaluating {total}[/cyan]\n")

    # ── Load checkpoint ──
    checkpoint = load_checkpoint(args.checkpoint) if args.resume else {
        "results": {}, "errors": [], "last_index": -1
    }
    start_index = checkpoint["last_index"] + 1 if args.resume else 0
    if args.resume and checkpoint["results"]:
        console.print(
            f"[green]📌 Resuming: {len(checkpoint['results'])} essays done, "
            f"starting at index {start_index}[/green]\n"
        )

    # ── Ensure Ollama is running ──
    console.print(f"[cyan]🤖 Starting Ollama with model: [bold]{args.model}[/bold]...[/cyan]")
    if not ensure_ollama_running(max_wait=30):
        console.print("[red]✗ Cannot start Ollama. Exiting.[/red]")
        sys.exit(1)

    llm = ChatOllama(
        model=args.model,
        temperature=args.temperature,
        num_predict=args.num_predict,
        num_ctx=args.num_ctx,
        keep_alive="30m",
    )

    # Quick smoke-test
    try:
        llm.invoke([HumanMessage(content="hi")])
        console.print("[green]✓ Model loaded![/green]\n")
    except Exception as e:
        console.print(f"[red]✗ Model test failed: {e}[/red]")
        sys.exit(1)

    # ── Evaluate ──
    failed_count = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating essays", total=total, completed=start_index)

        for i in range(start_index, total):
            essay_id = f"{DATASET_PREFIX}_{i}"

            # Skip already-done
            if str(i) in checkpoint["results"]:
                progress.advance(task)
                continue

            essay_data = essays[i]
            prompt_text = essay_data.get("prompt", "").strip()
            essay_text = essay_data.get("essay", "").strip()

            if not essay_text:
                console.print(f"[yellow]  ⚠ {essay_id}: Empty essay → 5.0[/yellow]")
                checkpoint["results"][str(i)] = 5.0
                checkpoint["last_index"] = i
                save_checkpoint(args.checkpoint, checkpoint)
                progress.advance(task)
                continue

            try:
                score, raw = evaluate_essay(llm, prompt_text, essay_text)

                if score is None:
                    console.print(
                        f"[yellow]  ⚠ {essay_id}: parse failed → 5.0[/yellow]"
                    )
                    score = 5.0
                    failed_count += 1
                    checkpoint["errors"].append({
                        "index": i,
                        "essay_id": essay_id,
                        "error": "score_parse_failed",
                        "response_preview": raw[:300],
                    })
                else:
                    all_scores = parse_all_scores(raw)
                    if all_scores:
                        s = " | ".join(f"{k}:{v}" for k, v in all_scores.items())
                        progress.console.print(f"  [dim]#{i} → {s}[/dim]")

                checkpoint["results"][str(i)] = score
                checkpoint["last_index"] = i

                # Checkpoint every essay (for reliability)
                save_checkpoint(args.checkpoint, checkpoint)

            except Exception as e:
                console.print(f"[red]  ✗ {essay_id}: {e}[/red]")
                checkpoint["results"][str(i)] = 5.0
                checkpoint["errors"].append({
                    "index": i,
                    "essay_id": essay_id,
                    "error": str(e),
                })
                failed_count += 1
                checkpoint["last_index"] = i
                save_checkpoint(args.checkpoint, checkpoint)

            progress.advance(task)

    # ── Write output ──
    write_output_csv(args.output, checkpoint["results"], total)
    console.print(f"\n[bold green]✓ Results saved to: {args.output}[/bold green]")
    console.print(f"[cyan]  Evaluated: {len(checkpoint['results'])} essays[/cyan]")
    console.print(f"[cyan]  Parse failures: {failed_count}[/cyan]")

    # Preview
    console.print(f"\n[bold]📋 Output preview:[/bold]")
    with open(args.output) as f:
        for i, line in enumerate(f):
            if i >= 6:
                break
            console.print(f"  {line.strip()}")

    console.print(f"\n[bold green]🎉 Done![/bold green]\n")


if __name__ == "__main__":
    main()
