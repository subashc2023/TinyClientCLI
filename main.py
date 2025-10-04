import asyncio
import json
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pyfiglet import figlet_format
from rich.console import Console
from rich.text import Text

# ============================================================================
# Config & Constants
# ============================================================================

console = Console()

DEFAULT_SYSTEM = (
    "You are TinyClient, a powerful, tiny MCP client and helpful assistant. "
    "You can call tools sequentially (waiting for each result before the next call) "
    "or simultaneously (multiple tools in parallel when the operations are independent)."
)

COLORS = {
    "user": "blue",
    "assistant": "green",
    "tool_call": "yellow",
    "tool_result": "purple",
    "error": "red",
}


@dataclass
class AppConfig:
    # Files / runtime
    config_file: Path = Path(__file__).parent / "mcp_config.json"

    # LLM settings
    system_prompt: str = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM)
    model: str = os.getenv("MODEL", "gpt-4o")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    base_url: str = os.getenv("BASE_URL", "https://api.openai.com")

    # Behavior knobs
    max_history_messages: int = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))
    max_tool_chars: int = int(os.getenv("MAX_TOOL_CHARS", "8000"))
    max_parallel_tools: int = int(os.getenv("MAX_PARALLEL_TOOLS", "8"))

    # HTTP
    request_timeout: float = float(os.getenv("HTTP_TIMEOUT_SECONDS", "30"))
    read_timeout: float = float(os.getenv("HTTP_READ_TIMEOUT_SECONDS", "60"))
    max_connections: int = int(os.getenv("HTTP_MAX_CONNECTIONS", "10"))
    max_keepalive: int = int(os.getenv("HTTP_MAX_KEEPALIVE", "5"))


# ============================================================================
# Console Helpers (DRY)
# ============================================================================

class UI:
    def __init__(self, console: Console):
        self.console = console

    def _safe_width(self, minus: int = 2) -> int:
        return max(self.console.width - minus, 10)

    def delimiter(self, color: str) -> None:
        width = self._safe_width(2)
        self.console.print(f"[{color}]│{'─' * width}[/{color}]")

    def margin_print(self, text: str, color: str, markup: bool = True) -> None:
        for line in text.splitlines():
            if markup:
                self.console.print(f"[{color}]│[/{color}] {line}")
            else:
                border = Text("│ ", style=color)
                content = Text(line)
                self.console.print(border + content)

    def margin_label(self, label: str, color: str) -> None:
        self.console.print(f"[bold {color}]│[/bold {color}] [bold]{label}:[/bold]")

    def section_block(self, color: str, header: Optional[str] = None) -> None:
        inner = self._safe_width(4)
        self.console.print(f"[{color}]│{'─' * inner}[/{color}]")
        if header:
            self.console.print(f"[{color}]│[/{color}] [bold]{header}[/bold]")


# ============================================================================
# MCP Server Management
# ============================================================================

class MCPManager:
    """Boot MCP servers from config; expose namespaced tool registry and invocations."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.servers: Dict[str, ClientSession] = {}
        self.transports: Dict[str, Any] = {}
        # tool_name -> (server_name, original_tool_name, openai_tool_schema)
        self.tools: Dict[str, Tuple[str, str, dict]] = {}

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_):
        await self.stop()

    async def start(self) -> None:
        cfg_path = self.config.config_file
        if not cfg_path.exists():
            console.print(f"[red]Config not found:[/red] {cfg_path}")
            return
        try:
            config_json = json.loads(cfg_path.read_text())
        except Exception as e:
            console.print(f"[red]Invalid config JSON:[/red] {e}")
            return

        for name, cfg in config_json.get("mcpServers", {}).items():
            transport_ctx = None
            session = None
            try:
                params = StdioServerParameters(
                    command=cfg["command"],
                    args=cfg.get("args", []),
                    env=cfg.get("env"),
                )
                transport_ctx = stdio_client(params)
                read, write = await transport_ctx.__aenter__()
                session = ClientSession(read, write)
                await session.__aenter__()
                await session.initialize()

                self.servers[name] = session
                self.transports[name] = transport_ctx

                # Discover tools and namespace them
                result = await session.list_tools()
                for tool in result.tools:
                    schema = tool.inputSchema or {"type": "object", "properties": {}}
                    if "additionalProperties" not in schema:
                        schema["additionalProperties"] = True
                    ns = f"{name}_{tool.name}"
                    self.tools[ns] = (
                        name,
                        tool.name,
                        {
                            "type": "function",
                            "function": {
                                "name": ns,
                                "description": tool.description or f"Tool: {tool.name}",
                                "parameters": schema,
                            },
                        },
                    )
            except Exception as e:
                console.print(f"[red]✗[/red] {name}: {e}")
                if session:
                    try:
                        await session.__aexit__(None, None, None)
                    except Exception:
                        pass
                if transport_ctx:
                    try:
                        await transport_ctx.__aexit__(None, None, None)
                    except Exception:
                        pass

    def server_summaries(self) -> List[Tuple[str, int]]:
        counts: Dict[str, int] = {}
        for server_name, _, _ in self.tools.values():
            counts[server_name] = counts.get(server_name, 0) + 1
        return list(counts.items())

    async def stop(self) -> None:
        for session in self.servers.values():
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                pass
        for transport in self.transports.values():
            try:
                await transport.__aexit__(None, None, None)
            except Exception:
                pass

    async def call_tool(self, namespaced: str, args: dict) -> str:
        server_name, original, _ = self.tools[namespaced]
        session = self.servers[server_name]
        result = await session.call_tool(original, arguments=args)
        if not result.content:
            return "Tool executed successfully (no output)"
        texts: List[str] = []
        for item in result.content:
            text = getattr(item, "text", None)
            if text is not None:
                texts.append(str(text))
            else:
                tname = getattr(item, "type", type(item).__name__)
                texts.append(f"[[{tname} content omitted]]")
        return "\n".join(texts) if texts else str(result.content)


# ============================================================================
# LLM Client Abstraction (keeps OpenAI-compatible bits isolated)
# ============================================================================

class OpenAICompatibleLLM:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(cfg.request_timeout, read=cfg.read_timeout),
            limits=httpx.Limits(
                max_connections=cfg.max_connections,
                max_keepalive_connections=cfg.max_keepalive,
            ),
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def stream_chat(self, *, messages: List[dict], tools: List[dict]) -> AsyncIterator[dict]:
        """Yield SSE chunks with retry/backoff for 429/5xx."""
        backoff = 0.5
        attempts = 5
        for attempt in range(attempts):
            try:
                headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
                payload = {"model": self.cfg.model, "stream": True, "messages": messages}
                if tools:
                    payload["tools"] = tools
                async with self.client.stream(
                    "POST",
                    f"{self.cfg.base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                ) as resp:
                    if resp.status_code in (429, 500, 502, 503, 504):
                        msg = (await resp.aread()).decode(errors="ignore")
                        raise RuntimeError(f"Retryable status {resp.status_code}: {msg}")
                    if resp.status_code != 200:
                        msg = (await resp.aread()).decode(errors="ignore")
                        raise RuntimeError(f"API Error {resp.status_code}: {msg}")
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                return
                            yield json.loads(data)
                    return
            except Exception:
                if attempt == attempts - 1:
                    raise
                await asyncio.sleep(backoff)
                backoff *= 2


# ============================================================================
# Chat Orchestrator (streaming, tool batches, message management)
# ============================================================================

class Chat:
    def __init__(self, cfg: AppConfig, mcp: MCPManager, llm: OpenAICompatibleLLM, ui: UI):
        self.cfg = cfg
        self.mcp = mcp
        self.llm = llm
        self.ui = ui
        self.messages: List[dict] = [{"role": "system", "content": cfg.system_prompt}]
        self.tool_call_counter = 0
        self.batch_counter = 0
        self.tool_sem = asyncio.Semaphore(cfg.max_parallel_tools)

    # ------------------------------
    # Internal helpers
    # ------------------------------

    def _trim_history(self) -> None:
        # Keep system + last N messages
        if len(self.messages) > self.cfg.max_history_messages + 1:
            self.messages = [self.messages[0]] + self.messages[-self.cfg.max_history_messages :]

    def _accumulate_tool_calls(self, delta_tool_calls: List[dict]) -> Dict[int, dict]:
        # Accumulate by index, concat arguments fragments
        acc: Dict[int, dict] = {}
        for tc in delta_tool_calls:
            idx = tc.get("index", 0)
            slot = acc.setdefault(idx, {})
            for key in ("id", "type"):
                if key in tc:
                    slot[key] = tc[key]
            func = tc.get("function")
            if func:
                fslot = slot.setdefault("function", {})
                if "name" in func:
                    fslot["name"] = func["name"]
                if "arguments" in func:
                    fslot["arguments"] = fslot.get("arguments", "") + func["arguments"]
        return acc

    async def _execute_tool(self, tc_id: str, name: str, arguments: str) -> Tuple[str, dict]:
        async with self.tool_sem:
            try:
                args = json.loads(arguments) if arguments else {}
            except Exception as e:
                return tc_id, {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": f"Error: invalid JSON for arguments: {e}",
                    "is_error": True,
                }
            try:
                result = await self.mcp.call_tool(name, args)
                content = (
                    (result[: self.cfg.max_tool_chars] + "\n…(truncated)")
                    if len(result) > self.cfg.max_tool_chars
                    else result
                )
                return tc_id, {"role": "tool", "tool_call_id": tc_id, "content": content}
            except Exception as e:
                return tc_id, {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": f"Error: {e}",
                    "is_error": True,
                }

    # ------------------------------
    # Public API
    # ------------------------------

    async def turn(self, user_input: str) -> None:
        self.messages.append({"role": "user", "content": user_input})
        self._trim_history()

        self.batch_counter = 0
        assistant_open = False
        first_chunk_of_content = True
        assistant_msg: dict = {"role": "assistant", "content": "", "tool_calls": []}

        while True:
            tool_calls_acc: Dict[int, dict] = {}
            async for chunk in self.llm.stream_chat(messages=self.messages, tools=[t[2] for t in self.mcp.tools.values()]):
                delta = chunk.get("choices", [{}])[0].get("delta", {})

                # Stream content
                content = delta.get("content")
                if content:
                    if not assistant_open:
                        self.ui.delimiter(COLORS["assistant"])
                        self.ui.margin_label("Assistant", COLORS["assistant"])
                        assistant_open = True
                    if first_chunk_of_content:
                        console.print(f"[{COLORS['assistant']}]│[/{COLORS['assistant']}] ", end="")
                        first_chunk_of_content = False
                    assistant_msg["content"] += content
                    if "\n" in content:
                        lines = content.split("\n")
                        for i, line in enumerate(lines):
                            if i > 0:
                                console.print()
                                console.print(f"[{COLORS['assistant']}]│[/{COLORS['assistant']}] ", end="")
                            console.print(line, end="")
                    else:
                        console.print(content, end="")

                # Accumulate tool_calls
                if delta.get("tool_calls"):
                    acc = self._accumulate_tool_calls(delta["tool_calls"])
                    for k, v in acc.items():
                        if k not in tool_calls_acc:
                            tool_calls_acc[k] = v
                        else:
                            # Merge defensively: preserve existing id/type/name if not in delta
                            for key in ("id", "type"):
                                if key in v:
                                    tool_calls_acc[k][key] = v[key]
                            if "function" in v:
                                func_slot = tool_calls_acc[k].setdefault("function", {})
                                if "name" in v["function"]:
                                    func_slot["name"] = v["function"]["name"]
                                if "arguments" in v["function"]:
                                    func_slot["arguments"] = func_slot.get("arguments", "") + v["function"]["arguments"]

            # If tool calls were requested, execute batch
            if tool_calls_acc:
                # Ensure a new line before the tool section if we were streaming text
                if not first_chunk_of_content:
                    console.print()

                self.batch_counter += 1

                # Fold tool_calls into the assistant message we just streamed
                for tc in tool_calls_acc.values():
                    assistant_msg.setdefault("tool_calls", []).append(
                        {
                            "id": tc.get("id", "unknown"),
                            "type": "function",
                            "function": tc.get("function", {}),
                        }
                    )
                self.messages.append(assistant_msg)

                # Pretty print tool calls with consistent numbering (by tool_call_id)
                inner = max(console.width - 4, 10)
                console.print(
                    f"[{COLORS['assistant']}]│[/{COLORS['assistant']}] "
                    f"[{COLORS['tool_call']}]│{'─' * inner}[/{COLORS['tool_call']}]"
                )
                console.print(
                    f"[{COLORS['assistant']}]│[/{COLORS['assistant']}] "
                    f"[{COLORS['tool_call']}]│[/{COLORS['tool_call']}] "
                    f"[bold]Tools (Batch {self.batch_counter}):[/bold]"
                )

                display_ids: Dict[str, int] = {}
                for i, tc in enumerate(tool_calls_acc.values(), 1):
                    tc_id = tc.get("id", "unknown")
                    display_ids[tc_id] = i
                    info = tc.get("function", {})
                    args_preview = info.get("arguments", "")
                    if len(args_preview) > 100:
                        args_preview = args_preview[:97] + "…"

                    console.print(
                        f"[{COLORS['assistant']}]│[/{COLORS['assistant']}] "
                        f"[{COLORS['tool_call']}]│[/{COLORS['tool_call']}] "
                        f"[{COLORS['tool_call']}][{i:02d}][/{COLORS['tool_call']}] "
                        f"{info.get('name','?')} {args_preview}"
                    )

                console.print(
                    f"[{COLORS['assistant']}]│[/{COLORS['assistant']}] "
                    f"[{COLORS['tool_call']}]│{'─' * inner}[/{COLORS['tool_call']}]"
                )

                # Validate and prepare parallel tasks (skip missing/unknown names)
                valid_tasks: List[asyncio.Task] = []
                valid_ids: set[str] = set()
                synthetic_errors: List[Tuple[str, dict]] = []

                for tc in tool_calls_acc.values():
                    tc_id = tc.get("id", "unknown")
                    name = tc.get("function", {}).get("name", "")
                    args = tc.get("function", {}).get("arguments", "")

                    if not name:
                        # Emit console error and synthesize a tool error message so the model isn't stuck
                        console.print(
                            f"[{COLORS['assistant']}]│[/{COLORS['assistant']}] "
                            f"[{COLORS['error']}]│[/{COLORS['error']}] "
                            f"[red][??][/red] Missing function name for tool call"
                        )
                        synthetic_errors.append(
                            (
                                tc_id,
                                {
                                    "role": "tool",
                                    "tool_call_id": tc_id,
                                    "content": "Error: Missing function name",
                                    "is_error": True,
                                },
                            )
                        )
                        continue

                    if name not in self.mcp.tools:
                        console.print(
                            f"[{COLORS['assistant']}]│[/{COLORS['assistant']}] "
                            f"[{COLORS['error']}]│[/{COLORS['error']}] "
                            f"[red][??][/red] Unknown tool name: '{name}'"
                        )
                        synthetic_errors.append(
                            (
                                tc_id,
                                {
                                    "role": "tool",
                                    "tool_call_id": tc_id,
                                    "content": f"Error: Unknown tool: '{name}'",
                                    "is_error": True,
                                },
                            )
                        )
                        continue

                    valid_tasks.append(self._execute_tool(tc_id, name, args))
                    valid_ids.add(tc_id)

                results: List[Tuple[str, dict]] = []
                if valid_tasks:
                    results = await asyncio.gather(*valid_tasks)

                # Merge synthetic errors so every requested tool_call_id has a tool message
                results = results + synthetic_errors

                # Results header
                console.print(
                    f"[{COLORS['assistant']}]│[/{COLORS['assistant']}] "
                    f"[{COLORS['tool_result']}]│{'─' * inner}[/{COLORS['tool_result']}]"
                )
                console.print(
                    f"[{COLORS['assistant']}]│[/{COLORS['assistant']}] "
                    f"[{COLORS['tool_result']}]│[/{COLORS['tool_result']}] "
                    f"[bold]Results:[/bold]"
                )

                # Render each result with the matching display index
                for tc_id, msg in results:
                    num = display_ids.get(tc_id, 0)
                    content = msg.get("content", "")
                    preview = content if len(content) <= 200 else content[:197] + "…"
                    is_err = msg.get("is_error", False)
                    for i, line in enumerate(preview.splitlines()):
                        color = COLORS["error"] if is_err else COLORS["tool_result"]
                        prefix = (
                            f"[{COLORS['assistant']}]│[/{COLORS['assistant']}] "
                            f"[{color}]│[/{color}] "
                        )
                        if i == 0:
                            tag = ("[red]" if is_err else f"[{COLORS['tool_result']}]") + f"[{num:02d}]" + (
                                "[/red]" if is_err else f"[/{COLORS['tool_result']}]"
                            )
                            console.print(prefix + tag + f" {line}")
                        else:
                            console.print(prefix + "     " + line)

                console.print(
                    f"[{COLORS['assistant']}]│[/{COLORS['assistant']}] "
                    f"[{COLORS['tool_result']}]│{'─' * inner}[/{COLORS['tool_result']}]"
                )

                # Append tool messages to history (IMPORTANT: only the dicts, not the (id, msg) tuples)
                tool_msgs = [msg for _, msg in results]
                self.messages.extend(tool_msgs)

                # Reset for potential further streaming/tool requests and continue loop
                first_chunk_of_content = True
                assistant_msg = {"role": "assistant", "content": "", "tool_calls": []}
                continue

            # No tool calls requested; finalize assistant message
            if not assistant_msg.get("tool_calls"):
                assistant_msg.pop("tool_calls", None)
            self.messages.append(assistant_msg)
            if assistant_open:
                console.print()
                self.ui.delimiter(COLORS["assistant"])
            break


# ============================================================================
# REPL
# ============================================================================

async def run_repl(cfg: AppConfig) -> None:
    ui = UI(console)
    async with MCPManager(cfg) as mcp:
        if not mcp.tools:
            console.print("[red]No tools available. Check config.[/red]")
            return
        # Banner
        width = max(console.width - 2, 10)
        console.print()
        console.print(f"[{COLORS['assistant']}]│{'─' * width}[/{COLORS['assistant']}]")
        for line in figlet_format("TinyClient", font="standard").splitlines():
            ui.margin_print(line, COLORS["assistant"], markup=False)
        ui.margin_print("A powerful, tiny MCP client for tool-enhanced conversations", COLORS["assistant"])
        for name, count in mcp.server_summaries():
            ui.margin_print(f"✓ {name}: {count} tools", COLORS["assistant"])
        ui.margin_print("Type 'exit' to quit, 'clear' to reset", COLORS["assistant"])
        console.print(f"[{COLORS['assistant']}]│{'─' * width}[/{COLORS['assistant']}]")

        llm = OpenAICompatibleLLM(cfg)
        chat = Chat(cfg, mcp, llm, ui)

        try:
            while True:
                ui.delimiter(COLORS["user"])
                ui.margin_label("You", COLORS["user"])
                console.print(f"[{COLORS['user']}]│[/{COLORS['user']}] ", end="")
                user_input = await asyncio.to_thread(console.input, "")
                if not user_input.strip():
                    continue
                if user_input.lower() in ("exit", "quit"):
                    break
                if user_input.lower() == "clear":
                    chat.messages = [chat.messages[0]]
                    chat.tool_call_counter = 0
                    chat.batch_counter = 0
                    ui.margin_print("Cleared", COLORS["user"])
                    ui.delimiter(COLORS["user"])
                    continue
                ui.delimiter(COLORS["user"])
                await chat.turn(user_input)
        finally:
            await llm.aclose()


# ============================================================================
# Entrypoint
# ============================================================================

async def main() -> None:
    cfg = AppConfig()
    stop = asyncio.Event()

    def handle_signal() -> None:
        stop.set()
        console.print("\n[yellow]Shutting down...[/yellow]")

    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT)):
        try:
            signal.signal(sig, lambda *_: handle_signal())
        except Exception:
            pass

    repl_task = asyncio.create_task(run_repl(cfg))
    stopper = asyncio.create_task(stop.wait())
    await asyncio.wait([repl_task, stopper], return_when=asyncio.FIRST_COMPLETED)
    if not repl_task.done():
        repl_task.cancel()
        try:
            await repl_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
