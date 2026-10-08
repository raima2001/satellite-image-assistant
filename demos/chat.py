"""Chat with the running agent from the terminal.

A thin client for POST /chat: it keeps one session id for the whole chat and
renders each reply as formatted Markdown (tables, bold, code) instead of raw
JSON. Start the API first (see README), then run:

    uv run python demos/chat.py                 # new random session id
    uv run python demos/chat.py --session demo-3

Type /trace to print this session's trace file, /exit (or Ctrl+C) to quit.
"""

import argparse
import json
from pathlib import Path
from uuid import uuid4

import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

TRACE_DIR = Path(__file__).resolve().parent.parent / "traces"

console = Console()


def show_trace(session_id: str) -> None:
    trace_file = TRACE_DIR / f"{session_id}.jsonl"
    if not trace_file.exists():
        console.print(f"[yellow]No trace yet at {trace_file}[/yellow]")
        return
    for line in trace_file.read_text().splitlines():
        record = json.loads(line)
        parts = [f"[bold]{record['step']}[/bold]"]
        if record.get("tool_name"):
            parts.append(f"tool={record['tool_name']} args={json.dumps(record['arguments'])}")
        if record.get("skill"):
            parts.append(f"skill={record['skill']}")
        if record.get("error"):
            parts.append(f"[red]error={record['error']}[/red]")
        if record.get("final_answer"):
            parts.append("[green]final answer[/green]")
        console.print("  " + "  ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", default=f"chat-{uuid4().hex[:8]}")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    console.print(
        f"Session [bold]{args.session}[/bold] - "
        "type /trace to see the trace, /exit to quit.\n"
    )
    with httpx.Client(base_url=args.url, timeout=180.0) as client:
        while True:
            try:
                message = console.input("[bold cyan]You:[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not message:
                continue
            if message == "/exit":
                break
            if message == "/trace":
                show_trace(args.session)
                continue

            try:
                with console.status("Agent is working..."):
                    response = client.post(
                        "/chat", json={"session_id": args.session, "message": message}
                    )
                reply = response.json()["reply"]
            except httpx.ConnectError:
                console.print(f"[red]Cannot reach the API at {args.url}. Is it running?[/red]")
                continue
            except (httpx.HTTPError, ValueError, KeyError) as error:
                console.print(f"[red]Request failed: {error}[/red]")
                continue

            console.print(Panel(Markdown(reply), title="Agent", border_style="green"))
    console.print(f"\nTrace saved at traces/{args.session}.jsonl")


if __name__ == "__main__":
    main()
