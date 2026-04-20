"""
agent/cli.py
────────────
IELTSCLI — interactive CLI for the multi-agent IELTS Writing Tutor system.

Orchestrates two agents via the supervisor graph:
  • Examiner Agent  — objective scoring with HITL review + self-critique
  • Tutor Agent     — challenge-and-revise + personalised lesson planning

Evaluation flow (human perspective)
────────────────────────────────────
  1. Paste task prompt + essay
  2. ⚙️  Phase 1: plan → tools run automatically
  3. 📋 Human reviews tool analysis (word count, grammar scan, keywords)
       → Confirm or cancel before any score is given
  4. 🎓 Phase 2: Examiner scores → Tutor reviews → (optional challenge)
                 → Final agreed score + personalised lesson plan
"""

from __future__ import annotations

import sys

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from agent.display import (
    display_evaluation,
    display_score_table,
    display_tool_summary,
    display_tutor_feedback,
    parse_scores,
)
from agent.checkpointing import build_checkpointer
from agent.schemas import IELTSEvaluation
from agent.supervisor import SupervisorState, build_supervisor_graph
from agent.prompts import COMMANDS

console = Console()

# Default initial state for every graph invocation
_INIT_STATE: dict = {
    "is_eval":           False,
    "tool_results":      {},
    "evaluation":        {},
    "critique_feedback": "",
    "revision_count":    0,
    "tutor_challenge":   "",
    "examiner_verdict":  "",
    "tutor_feedback":    {},
}


class IELTSCLI:
    """
    Interactive CLI for the multi-agent IELTS Writing Tutor.

    Parameters
    ----------
    model         : Ollama model tag (used by both agents)
    examiner_temp : Examiner sampling temperature (default 0.1 — deterministic scoring)
    tutor_temp    : Tutor sampling temperature (default 0.6 — creative lesson plans)
    thread_id     : Checkpoint session key — unique per student
    postgres_uri  : Optional PostgreSQL URI for persistent LangGraph memory
    """

    def __init__(
        self,
        model: str,
        examiner_temp: float = 0.1,
        tutor_temp: float    = 0.6,
        thread_id: str       = "default",
        postgres_uri: str | None = None,
    ) -> None:
        self.model = model
        self.thread_id = thread_id
        self.config = {"configurable": {"thread_id": thread_id}}
        self.last_scores: dict[str, float] = {}
        self.last_tutor_feedback: dict = {}
        self._checkpointer_handle = build_checkpointer(postgres_uri)

        console.print(
            f"[cyan]⚡ Building multi-agent graph "
            f"(Examiner[dim] T={examiner_temp}[/dim] · "
            f"Tutor[dim] T={tutor_temp}[/dim]) "
            f"for model [bold]{model}[/bold]…[/cyan]"
        )
        try:
            self.graph = build_supervisor_graph(
                model=model,
                examiner_temp=examiner_temp,
                tutor_temp=tutor_temp,
                checkpointer=self._checkpointer_handle.checkpointer,
            )
            console.print("[dim]  Testing connection…[/dim]")
            self._run_chat_turn("Hi")
            console.print(
                f"[dim]  Memory backend: [bold]{self._checkpointer_handle.backend}[/bold][/dim]"
            )
            console.print("[green]✓ Multi-agent system ready![/green]\n")
        except Exception as exc:
            console.print(f"[red]✗ Could not initialise: {exc}[/red]")
            console.print("[yellow]  ollama serve  →  start server[/yellow]")
            console.print(f"[yellow]  ollama run {model}  →  pull model[/yellow]")
            self._checkpointer_handle.close()
            sys.exit(1)

    def _shutdown(self) -> None:
        """Release checkpointer resources (e.g., Postgres connections)."""
        self._checkpointer_handle.close()

    # ══════════════════════════════════════════════════════════════════════════
    # Graph invocation helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _run_chat_turn(self, text: str) -> str:
        """Single-turn chat via supervisor graph (routes to END before Tutor)."""
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=text)], **_INIT_STATE, "is_eval": False},
            config=self.config,
        )
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content
        return ""

    def _eval_phase1(self, user_input: str) -> bool:
        """
        Phase 1: run plan + tools, stop at the HITL interrupt.
        Returns True if the graph is waiting at 'evaluate'.
        """
        self.graph.invoke(
            {"messages": [HumanMessage(content=user_input)], **_INIT_STATE, "is_eval": True},
            config=self.config,
        )
        state = self.graph.get_state(self.config)
        return "evaluate" in (state.next or ())

    def _eval_phase2(self) -> tuple[dict, dict]:
        """
        Phase 2: resume from interrupt → Examiner scores →
                 Tutor reviews → (optional challenge) → lesson plan.
        Returns (evaluation_dict, tutor_feedback_dict).
        """
        result = self.graph.invoke(None, config=self.config)
        return result.get("evaluation", {}), result.get("tutor_feedback", {})

    # ══════════════════════════════════════════════════════════════════════════
    # Command handlers
    # ══════════════════════════════════════════════════════════════════════════

    def handle_eval(self) -> None:
        """Full two-phase evaluation with HITL + multi-agent output."""
        console.print(Panel(
            "[bold cyan]ESSAY EVALUATION MODE[/bold cyan]  "
            "[dim]— Examiner + Tutor Agents[/dim]\n\n"
            "Paste your IELTS Task 2 [bold]prompt[/bold], then type [bold]END[/bold].\n"
            "Then paste your [bold]essay[/bold] and type [bold]END[/bold].\n\n"
            "[dim]Flow:  tools → human review → Examiner scores → "
            "Tutor challenge? → Lesson plan[/dim]",
            box=box.ROUNDED,
        ))

        console.print("\n[bold yellow]▶ Task Prompt (END to finish):[/bold yellow]")
        task_prompt = self._read_multiline()

        console.print("\n[bold yellow]▶ Your Essay (END to finish):[/bold yellow]")
        essay = self._read_multiline()

        if not essay:
            console.print("[red]No essay provided.[/red]")
            return

        user_input = (
            "Please evaluate this IELTS Writing Task 2 essay.\n\n"
            f"**Task Prompt:** {task_prompt}\n\n"
            f"**Essay:**\n{essay}"
        )

        # ── Phase 1: plan + tools ─────────────────────────────────────────────
        console.print("\n[bold cyan]⚙️  Phase 1 — Running tool analysis…[/bold cyan]\n")
        at_interrupt = self._eval_phase1(user_input)

        if not at_interrupt:
            console.print("[yellow]Graph ended early — no evaluation produced.[/yellow]")
            return

        # ── HITL review ───────────────────────────────────────────────────────
        tool_results = self.graph.get_state(self.config).values.get("tool_results", {})
        console.print()
        display_tool_summary(tool_results)
        console.print()

        confirmed = Confirm.ask(
            "[bold yellow]📋 Tool analysis complete. Proceed with Examiner + Tutor scoring?[/bold yellow]",
            default=True,
        )
        if not confirmed:
            console.print("[yellow]⏹  Evaluation cancelled.[/yellow]")
            return

        # ── Phase 2: Examiner → Tutor ─────────────────────────────────────────
        console.print(
            "\n[bold green]🎓 Phase 2 — Examiner scoring + Tutor lesson planning…"
            "[/bold green]\n"
        )
        evaluation, tutor_feedback = self._eval_phase2()

        # ── Display Examiner output ───────────────────────────────────────────
        if evaluation:
            display_evaluation(evaluation)
            try:
                ev_obj = IELTSEvaluation.model_validate(evaluation)
                self.last_scores = ev_obj.to_scores_dict()
            except Exception:
                self.last_scores = {}
        else:
            # Fallback: grab last AI message text
            state = self.graph.get_state(self.config)
            for msg in reversed(state.values.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    from rich.markdown import Markdown
                    console.print(Markdown(msg.content))
                    self.last_scores = parse_scores(msg.content)
                    break

        if self.last_scores:
            display_score_table(self.last_scores)

        # ── Display Tutor output ──────────────────────────────────────────────
        if tutor_feedback:
            self.last_tutor_feedback = tutor_feedback
            display_tutor_feedback(tutor_feedback)
        else:
            # Fallback: print last AI message if no structured feedback
            state = self.graph.get_state(self.config)
            msgs = state.values.get("messages", [])
            if msgs:
                last = msgs[-1]
                if isinstance(last, AIMessage) and last.content:
                    from rich.markdown import Markdown
                    console.print(Markdown(last.content))

    def handle_chat(self, user_input: str) -> None:
        """Free-form chat with the system (Examiner answers as tutor in chat mode)."""
        console.print("\n[bold green]🤖 Tutor:[/bold green]")
        response = self._run_chat_turn(user_input)
        console.print(response)
        console.print()

    def handle_command(self, cmd: str) -> bool:
        """Dispatch slash commands. Returns False to exit."""
        cmd = cmd.strip().lower()

        if cmd in ("/quit", "/exit"):
            console.print("[yellow]Goodbye! Keep writing! 👋[/yellow]")
            return False

        elif cmd == "/help":
            t = Table(title="Commands", box=box.SIMPLE)
            t.add_column("Command",     style="bold cyan")
            t.add_column("Description")
            for c, d in COMMANDS.items():
                t.add_row(c, d)
            console.print(t)

        elif cmd == "/eval":
            self.handle_eval()

        elif cmd == "/score":
            display_score_table(self.last_scores)

        elif cmd == "/tutor":
            display_tutor_feedback(self.last_tutor_feedback)

        elif cmd.startswith("/history"):
            parts = cmd.split()
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 6
            state = self.graph.get_state(self.config)
            msgs  = state.values.get("messages", [])[-n:]
            for m in msgs:
                role  = type(m).__name__
                color = "blue" if "Human" in role else "green" if "AI" in role else "cyan"
                preview = (m.content or "")[:300]
                console.print(f"[{color}]{role}:[/{color}] {preview}\n")

        else:
            console.print(f"[red]Unknown command: {cmd}. Type /help for available commands.[/red]")

        return True

    # ══════════════════════════════════════════════════════════════════════════
    # Main loop
    # ══════════════════════════════════════════════════════════════════════════

    def run(self) -> None:
        """Start the interactive CLI session."""
        console.print(Panel(
            "[bold cyan]🎓 IELTS Writing Tutor System[/bold cyan]  "
            "[dim]v3 — Multi-Agent Edition[/dim]\n\n"
            f"Model   : [bold]{self.model}[/bold]\n"
            f"Session : [bold]{self.thread_id}[/bold]\n"
            "Agents  : [bold cyan]Examiner[/bold cyan] + [bold green]Tutor[/bold green]\n\n"
            "Type [bold]/eval[/bold] to evaluate an essay  •  "
            "[bold]/help[/bold] for all commands  •  "
            "[bold]/tutor[/bold] to re-show last lesson plan",
            box=box.DOUBLE,
            border_style="cyan",
        ))

        try:
            while True:
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]").strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    if not self.handle_command(user_input):
                        break
                    continue

                if len(user_input) > 250:
                    console.print(
                        "[yellow]Looks like a pasted essay — "
                        "use [bold]/eval[/bold] for structured scoring.[/yellow]"
                    )

                self.handle_chat(user_input)
        except KeyboardInterrupt:
            console.print("\n[yellow]Use /quit to exit.[/yellow]")
        except EOFError:
            pass
        finally:
            self._shutdown()

    @staticmethod
    def _read_multiline() -> str:
        """Read stdin lines until 'END'."""
        lines: list[str] = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        return "\n".join(lines).strip()
