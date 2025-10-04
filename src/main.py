import asyncio
import json
import logging
import os
import signal
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pyfiglet import figlet_format
from rich.console import Console
from rich.text import Text

from types_def import Message, ToolCall, ToolResult, Role


DEFAULT_SYSTEM = (
    "You are TinyClient, a powerful, tiny MCP client and helpful assistant. "
    "You can call tools sequentially (waiting for each result before the next call) "
    "or simultaneously (multiple tools in parallel when the operations are independent)."
)

TOOL_ARGS_PREVIEW_LEN = 100
TOOL_RESULT_PREVIEW_LEN = 200

logger = logging.getLogger(__name__)


class Color(Enum):
    USER = "blue"
    ASSISTANT = "green"
    TOOL_CALL = "yellow"
    TOOL_RESULT = "purple"
    ERROR = "red"
    INFO = "cyan"



@dataclass
class AppConfig:
    # Files / runtime
    config_file: Path = field(default_factory=lambda: Path(__file__).parent / "mcp_config.json")

    # LLM settings
    system_prompt: str = field(default_factory=lambda: os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM))
    model: str = field(default_factory=lambda: os.getenv("MODEL", "gpt-4o"))
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("BASE_URL", "https://api.openai.com"))

    # Behavior knobs
    max_history_messages: int = field(default_factory=lambda: int(os.getenv("MAX_HISTORY_MESSAGES", "20")))
    max_tool_chars: int = field(default_factory=lambda: int(os.getenv("MAX_TOOL_CHARS", "8000")))
    max_parallel_tools: int = field(default_factory=lambda: int(os.getenv("MAX_PARALLEL_TOOLS", "8")))
    stream_tool_results: bool = field(default_factory=lambda: os.getenv("STREAM_TOOL_RESULTS", "false").lower() == "true")

    # HTTP retry settings
    request_timeout: float = field(default_factory=lambda: float(os.getenv("HTTP_TIMEOUT_SECONDS", "30")))
    read_timeout: float = field(default_factory=lambda: float(os.getenv("HTTP_READ_TIMEOUT_SECONDS", "60")))
    max_connections: int = field(default_factory=lambda: int(os.getenv("HTTP_MAX_CONNECTIONS", "10")))
    max_keepalive: int = field(default_factory=lambda: int(os.getenv("HTTP_MAX_KEEPALIVE", "5")))
    retry_attempts: int = field(default_factory=lambda: int(os.getenv("HTTP_RETRY_ATTEMPTS", "5")))
    retry_backoff_base: float = field(default_factory=lambda: float(os.getenv("HTTP_RETRY_BACKOFF", "0.5")))

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "WARNING"))
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.max_parallel_tools < 1:
            raise ValueError("max_parallel_tools must be >= 1")
        if self.max_history_messages < 1:
            raise ValueError("max_history_messages must be >= 1")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        if self.retry_attempts < 1:
            raise ValueError("retry_attempts must be >= 1")

        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )



class UI:
    
    def __init__(self, console: Console):
        self.console = console
    
    def _safe_width(self, minus: int = 2) -> int:
        return max(self.console.width - minus, 10)

    def prefix(self, role: Color) -> str:
        return f"[{role.value}]│[/{role.value}] "

    def nested_prefix(self, outer: Color, inner: Color) -> str:
        return f"{self.prefix(outer)}[{inner.value}]│[/{inner.value}] "

    def border(self, color: Color, width: Optional[int] = None, char: str = '─') -> str:
        w = width or self._safe_width(2)
        return f"[{color.value}]│{char * w}[/{color.value}]"

    def delimiter(self, color: Color) -> None:
        self.console.print(self.border(color))

    def section_header(self, color: Color, text: str) -> None:
        inner_width = self._safe_width(4)
        self.console.print(f"[{color.value}]│{inner_width * '─'}[/{color.value}]")
        self.console.print(f"{self.prefix(color)}[bold]{text}[/bold]")

    def label(self, color: Color, text: str) -> None:
        self.console.print(f"[bold {color.value}]│[/bold {color.value}] [bold]{text}:[/bold]")

    def text(self, color: Color, text: str, markup: bool = True) -> None:
        for line in text.splitlines():
            if markup:
                self.console.print(f"{self.prefix(color)}{line}")
            else:
                border = Text("│ ", style=color.value)
                content = Text(line)
                self.console.print(border + content)
    
    def input_prompt(self, color: Color) -> None:
        self.console.print(f"{self.prefix(color)}", end="")

    def banner(self, color: Color, title: str, subtitle: str, info_lines: List[str]) -> None:
        width = self._safe_width()
        self.console.print()
        self.console.print(self.border(color, width))
        for line in figlet_format(title, font="standard").splitlines():
            self.text(color, line, markup=False)
        self.text(color, subtitle)
        for line in info_lines:
            self.text(color, line)
        self.console.print(self.border(color, width))

    def tool_batch_header(self, outer: Color, inner: Color, batch_num: int) -> None:
        inner_width = self._safe_width(4)
        self.console.print(
            f"{self.prefix(outer)}{self.border(inner, inner_width)}"
        )
        self.console.print(
            f"{self.nested_prefix(outer, inner)}[bold]Tools (Batch {batch_num}):[/bold]"
        )

    def tool_call_item(self, outer: Color, inner: Color, index: int, name: str, args_preview: str) -> None:
        self.console.print(
            f"{self.nested_prefix(outer, inner)}"
            f"[{inner.value}][{index:02d}][/{inner.value}] {name} {args_preview}"
        )

    def tool_batch_footer(self, outer: Color, inner: Color) -> None:
        inner_width = self._safe_width(4)
        self.console.print(
            f"{self.prefix(outer)}{self.border(inner, inner_width)}"
        )

    def tool_results_header(self, outer: Color, inner: Color) -> None:
        inner_width = self._safe_width(4)
        self.console.print(
            f"{self.prefix(outer)}{self.border(inner, inner_width)}"
        )
        self.console.print(
            f"{self.nested_prefix(outer, inner)}[bold]Results:[/bold]"
        )

    def tool_result_item(self, outer: Color, index: int, content: str, is_error: bool = False) -> None:
        color = Color.ERROR if is_error else Color.TOOL_RESULT
        lines = content.splitlines()
        for i, line in enumerate(lines):
            prefix = self.nested_prefix(outer, color)
            if i == 0:
                tag = f"[{'red' if is_error else color.value}][{index:02d}][/{'red' if is_error else color.value}]"
                self.console.print(f"{prefix}{tag} {line}")
            else:
                self.console.print(f"{prefix}     {line}")

    def error(self, outer: Color, message: str) -> None:
        self.console.print(
            f"{self.nested_prefix(outer, Color.ERROR)}"
            f"[red][??][/red] {message}"
        )



class MCPManager:
    
    def __init__(self, config: AppConfig, console: Console):
        self.config = config
        self.console = console
        self.servers: Dict[str, ClientSession] = {}
        self.transports: Dict[str, Any] = {}
        self.tools: Dict[str, Tuple[str, str, dict]] = {}
    
    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_):
        await self.stop()

    async def start(self) -> None:
        cfg_path = self.config.config_file
        if not cfg_path.exists():
            logger.error(f"Config not found: {cfg_path}")
            self.console.print(f"[red]Config not found:[/red] {cfg_path}")
            return

        try:
            config_json = json.loads(cfg_path.read_text())
        except Exception as e:
            logger.error(f"Invalid config JSON: {e}", exc_info=True)
            self.console.print(f"[red]Invalid config JSON:[/red] {e}")
            return

        for name, cfg in config_json.get("mcpServers", {}).items():
            await self._initialize_server(name, cfg)

    async def _initialize_server(self, name: str, cfg: dict) -> None:
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

            await self._register_tools(name, session)
            logger.info(f"Initialized MCP server: {name}")

        except Exception as e:
            logger.error(f"Failed to initialize {name}: {e}", exc_info=True)
            self.console.print(f"[red]✗[/red] {name}: {e}")
            await self._cleanup_failed_server(session, transport_ctx)

    async def _register_tools(self, server_name: str, session: ClientSession) -> None:
        result = await session.list_tools()
        for tool in result.tools:
            schema = tool.inputSchema or {"type": "object", "properties": {}}
            if "additionalProperties" not in schema:
                schema["additionalProperties"] = True
            
            namespaced = f"{server_name}_{tool.name}"
            self.tools[namespaced] = (
                server_name,
                tool.name,
                {
                    "type": "function",
                    "function": {
                        "name": namespaced,
                        "description": tool.description or f"Tool: {tool.name}",
                        "parameters": schema,
                    },
                },
            )
    
    async def _cleanup_failed_server(self, session: Optional[ClientSession], transport_ctx: Optional[Any]) -> None:
        if session:
            try:
                await session.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error cleaning up session: {e}")
        if transport_ctx:
            try:
                await transport_ctx.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error cleaning up transport: {e}")

    def server_summaries(self) -> List[Tuple[str, int]]:
        counts: Dict[str, int] = {}
        for server_name, _, _ in self.tools.values():
            counts[server_name] = counts.get(server_name, 0) + 1
        return list(counts.items())

    async def stop(self) -> None:
        for name, session in self.servers.items():
            try:
                await session.__aexit__(None, None, None)
                logger.info(f"Stopped MCP server: {name}")
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")

        for transport in self.transports.values():
            try:
                await transport.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error stopping transport: {e}")

    async def call_tool(self, namespaced: str, args: dict) -> str:
        server_name, original, _ = self.tools[namespaced]
        session = self.servers.get(server_name)
        
        if not session:
            raise RuntimeError(f"MCP server '{server_name}' is unavailable")
        
        try:
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
            
        except Exception as e:
            logger.error(f"Tool execution failed for {namespaced}: {e}", exc_info=True)
            raise



class OpenAICompatibleLLM:
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.request_timeout, read=config.read_timeout),
            limits=httpx.Limits(
                max_connections=config.max_connections,
                max_keepalive_connections=config.max_keepalive,
            ),
        )
    
    async def aclose(self) -> None:
        await self.client.aclose()

    async def stream_chat(self, messages: List[Message], tools: List[dict]) -> AsyncIterator[dict]:
        backoff = self.config.retry_backoff_base

        for attempt in range(self.config.retry_attempts):
            try:
                async for chunk in self._stream_request(messages, tools):
                    yield chunk
                return

            except Exception as e:
                if attempt == self.config.retry_attempts - 1:
                    logger.error(f"All retry attempts exhausted: {e}", exc_info=True)
                    raise

                logger.warning(f"Retry attempt {attempt + 1}/{self.config.retry_attempts}: {e}")
                await asyncio.sleep(backoff)
                backoff *= 2

    async def _stream_request(self, messages: List[Message], tools: List[dict]) -> AsyncIterator[dict]:
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        payload = {
            "model": self.config.model,
            "stream": True,
            "messages": messages
        }
        if tools:
            payload["tools"] = tools
        
        async with self.client.stream(
            "POST",
            f"{self.config.base_url}/v1/chat/completions",
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
                if not line or not line.startswith("data: "):
                    continue
                
                data = line[6:]
                if data == "[DONE]":
                    return
                
                yield json.loads(data)



class Chat:
    
    def __init__(
        self,
        config: AppConfig,
        mcp: MCPManager,
        llm: OpenAICompatibleLLM,
        ui: UI
    ):
        self.config = config
        self.mcp = mcp
        self.llm = llm
        self.ui = ui
        self.messages: List[Message] = [
            {"role": "system", "content": config.system_prompt}
        ]
        self.batch_counter = 0
        self.tool_sem = asyncio.Semaphore(config.max_parallel_tools)

    async def turn(self, user_input: str) -> None:
        self._add_user_message(user_input)

        while True:
            assistant_msg = await self._stream_response()

            if assistant_msg.get("tool_calls"):
                await self._handle_tool_batch(assistant_msg)
            else:
                self._finalize_message(assistant_msg)
                break

    def _add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self._trim_history()

    def _trim_history(self) -> None:
        max_msgs = self.config.max_history_messages + 1
        if len(self.messages) > max_msgs:
            self.messages = [self.messages[0]] + self.messages[-self.config.max_history_messages:]

    async def _stream_response(self) -> Message:
        assistant_msg: Message = {"role": "assistant", "content": ""}
        tool_calls_acc: Dict[int, dict] = {}
        assistant_open = False
        first_chunk = True
        
        tools_schema = [schema for _, _, schema in self.mcp.tools.values()]
        
        async for chunk in self.llm.stream_chat(self.messages, tools_schema):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            
            content = delta.get("content")
            if content:
                if not assistant_open:
                    self.ui.delimiter(Color.ASSISTANT)
                    self.ui.label(Color.ASSISTANT, "Assistant")
                    assistant_open = True
                
                if first_chunk:
                    self.ui.input_prompt(Color.ASSISTANT)
                    first_chunk = False
                
                assistant_msg["content"] += content
                self._stream_content_chunk(content)
            
            if delta.get("tool_calls"):
                self._accumulate_tool_calls(delta["tool_calls"], tool_calls_acc)
        
        # Finalize streaming and tool calls
        if tool_calls_acc:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.get("id", "unknown"),
                    "type": "function",
                    "function": tc.get("function", {}),
                }
                for tc in tool_calls_acc.values()
            ]
        
        # Close assistant section cleanly
        if assistant_open or tool_calls_acc:
            if assistant_open and not first_chunk:
                self.ui.console.print()  # End the streaming line
            if assistant_open and not tool_calls_acc:
                self.ui.delimiter(Color.ASSISTANT)  # Only add delimiter if no tools follow
        
        return assistant_msg
    
    def _stream_content_chunk(self, content: str) -> None:
        if "\n" in content:
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if i > 0:
                    self.ui.console.print()
                    self.ui.input_prompt(Color.ASSISTANT)
                self.ui.console.print(line, end="")
        else:
            self.ui.console.print(content, end="")

    def _accumulate_tool_calls(self, delta_tool_calls: List[dict], acc: Dict[int, dict]) -> None:
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

    async def _handle_tool_batch(self, assistant_msg: Message) -> None:
        self.batch_counter += 1
        tool_calls = assistant_msg["tool_calls"]

        self.messages.append(assistant_msg)

        display_ids = self._display_tool_calls(tool_calls)

        if self.config.stream_tool_results:
            results = await self._execute_tool_batch_streaming(tool_calls, display_ids)
        else:
            results = await self._execute_tool_batch(tool_calls, display_ids)
            self._display_tool_results(results, display_ids)

        # Check if all tools failed
        if not results:
            self._handle_all_tools_failed()
            return

        tool_messages = [msg for _, msg in results]
        self.messages.extend(tool_messages)
        self._trim_history()

    def _handle_all_tools_failed(self) -> None:
        """Handle case where all tool calls failed validation."""
        available_tools = ", ".join(sorted(self.mcp.tools.keys())[:10])
        if len(self.mcp.tools) > 10:
            available_tools += f", ... ({len(self.mcp.tools) - 10} more)"
        
        fallback = {
            "role": "assistant",
            "content": f"All requested tool calls were invalid. Available tools: {available_tools}"
        }
        self.messages.append(fallback)
        self.ui.error(Color.ASSISTANT, "All tool calls failed validation - check available tools")

    def _display_tool_calls(self, tool_calls: List[ToolCall]) -> Dict[str, int]:
        self.ui.tool_batch_header(Color.ASSISTANT, Color.TOOL_CALL, self.batch_counter)
        
        display_ids: Dict[str, int] = {}
        for i, tc in enumerate(tool_calls, 1):
            tc_id = tc.get("id", "unknown")
            display_ids[tc_id] = i
            
            func = tc.get("function", {})
            name = func.get("name", "?")
            args = func.get("arguments", "")
            
            args_preview = args
            if len(args_preview) > TOOL_ARGS_PREVIEW_LEN:
                args_preview = args_preview[:TOOL_ARGS_PREVIEW_LEN - 1] + "…"
            
            self.ui.tool_call_item(Color.ASSISTANT, Color.TOOL_CALL, i, name, args_preview)
        
        self.ui.tool_batch_footer(Color.ASSISTANT, Color.TOOL_CALL)
        return display_ids
    
    async def _execute_tool_batch(self, tool_calls: List[ToolCall], display_ids: Dict[str, int]) -> List[Tuple[str, Message]]:
        """Execute tools in parallel and return all results."""
        tasks: List[asyncio.Task] = []
        sync_results: List[Tuple[str, Message]] = []

        for tc in tool_calls:
            tc_id = tc.get("id", "unknown")
            func = tc.get("function", {})
            name = func.get("name", "")
            args = func.get("arguments", "")

            error_msg = self._validate_tool_call(tc_id, name, display_ids)
            if error_msg:
                sync_results.append((tc_id, error_msg))
                continue

            tasks.append(asyncio.create_task(self._execute_single_tool(tc_id, name, args)))

        async_results = await asyncio.gather(*tasks) if tasks else []
        return sync_results + async_results

    async def _execute_tool_batch_streaming(self, tool_calls: List[ToolCall], display_ids: Dict[str, int]) -> List[Tuple[str, Message]]:
        """Execute tools and display results as they complete."""
        tasks_map: Dict[asyncio.Task, str] = {}
        sync_results: List[Tuple[str, Message]] = []

        for tc in tool_calls:
            tc_id = tc.get("id", "unknown")
            func = tc.get("function", {})
            name = func.get("name", "")
            args = func.get("arguments", "")

            error_msg = self._validate_tool_call(tc_id, name, display_ids)
            if error_msg:
                sync_results.append((tc_id, error_msg))
                continue

            task = asyncio.create_task(self._execute_single_tool(tc_id, name, args))
            tasks_map[task] = tc_id

        # Display results header
        if tasks_map or sync_results:
            self.ui.tool_results_header(Color.ASSISTANT, Color.TOOL_RESULT)

        # Display sync results first
        for tc_id, msg in sync_results:
            index = display_ids.get(tc_id, 0)
            content = msg.get("content", "")
            is_error = "Error:" in content
            preview = self._preview_content(content)
            self.ui.tool_result_item(Color.ASSISTANT, index, preview, is_error)

        # Stream async results as they complete
        results = sync_results.copy()
        for coro in asyncio.as_completed(tasks_map.keys()):
            tc_id, msg = await coro
            results.append((tc_id, msg))
            
            index = display_ids.get(tc_id, 0)
            content = msg.get("content", "")
            is_error = "Error:" in content
            preview = self._preview_content(content)
            self.ui.tool_result_item(Color.ASSISTANT, index, preview, is_error)

        if tasks_map or sync_results:
            self.ui.tool_batch_footer(Color.ASSISTANT, Color.TOOL_RESULT)

        return results

    def _validate_tool_call(self, tc_id: str, name: str, display_ids: Dict[str, int]) -> Optional[Message]:
        """Validate tool call and return error message if invalid."""
        if not name:
            self.ui.error(Color.ASSISTANT, "Missing function name for tool call")
            return self._create_error_message(tc_id, "Missing function name")

        if name not in self.mcp.tools:
            self.ui.error(Color.ASSISTANT, f"Unknown tool name: '{name}'")
            return self._create_error_message(tc_id, f"Unknown tool: '{name}'")

        return None

    def _create_error_message(self, tc_id: str, error: str) -> Message:
        """Create error tool message."""
        return {
            "role": "tool",
            "tool_call_id": tc_id,
            "content": f"Error: {error}",
        }

    async def _execute_single_tool(self, tc_id: str, name: str, arguments: str) -> Tuple[str, Message]:
        """Execute single tool with semaphore limiting."""
        async with self.tool_sem:
            try:
                args = json.loads(arguments) if arguments else {}
            except Exception as e:
                logger.warning(f"Invalid JSON arguments for {name}: {e}")
                return tc_id, self._create_error_message(tc_id, f"Invalid JSON: {e}")

            try:
                result = await self.mcp.call_tool(name, args)
                content = self._truncate_content(result)
                return tc_id, {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": content
                }
            except Exception as e:
                logger.error(f"Tool execution failed for {name}: {e}")
                return tc_id, self._create_error_message(tc_id, str(e))

    def _truncate_content(self, content: str) -> str:
        """Intelligently truncate content at natural boundaries."""
        max_chars = self.config.max_tool_chars
        if len(content) <= max_chars:
            return content
        
        # Try to truncate at natural boundary (newline)
        truncated = content[:max_chars]
        last_newline = truncated.rfind('\n')
        
        # If newline is in last 20% of allowed chars, truncate there
        if last_newline > max_chars * 0.8:
            truncated = content[:last_newline]
        
        # Calculate what was lost
        remaining = content[len(truncated):]
        chars_lost = len(remaining)
        lines_lost = remaining.count('\n')
        
        return f"{truncated}\n\n...(truncated {chars_lost} chars, ~{lines_lost} lines)"

    def _preview_content(self, content: str) -> str:
        """Create preview of content for display."""
        if len(content) <= TOOL_RESULT_PREVIEW_LEN:
            return content
        return content[:TOOL_RESULT_PREVIEW_LEN - 1] + "…"

    def _display_tool_results(self, results: List[Tuple[str, Message]], display_ids: Dict[str, int]) -> None:
        """Display all tool results at once."""
        self.ui.tool_results_header(Color.ASSISTANT, Color.TOOL_RESULT)

        for tc_id, msg in results:
            index = display_ids.get(tc_id, 0)
            content = msg.get("content", "")
            is_error = "Error:" in content
            preview = self._preview_content(content)
            self.ui.tool_result_item(Color.ASSISTANT, index, preview, is_error)

        self.ui.tool_batch_footer(Color.ASSISTANT, Color.TOOL_RESULT)

    def _finalize_message(self, assistant_msg: Message) -> None:
        """Add final assistant message to history and trim."""
        if not assistant_msg.get("tool_calls"):
            assistant_msg.pop("tool_calls", None)
        self.messages.append(assistant_msg)
        self._trim_history()

    def reset(self) -> None:
        """Reset conversation to initial state."""
        self.messages = [self.messages[0]]
        self.batch_counter = 0



async def run_repl(config: AppConfig, console: Console) -> None:
    ui = UI(console)
    
    async with MCPManager(config, console) as mcp:
        if not mcp.tools:
            console.print("[red]No tools available. Check config.[/red]")
            return
        
        info_lines = [f"✓ {name}: {count} tools" for name, count in mcp.server_summaries()]
        info_lines.append("Type 'exit' to quit, 'clear' to reset")
        ui.banner(
            Color.ASSISTANT,
            "TinyClient",
            "A powerful, tiny MCP client for tool-enhanced conversations",
            info_lines
        )

        llm = OpenAICompatibleLLM(config)
        chat = Chat(config, mcp, llm, ui)

        try:
            while True:
                ui.delimiter(Color.USER)
                ui.label(Color.USER, "You")
                ui.input_prompt(Color.USER)

                user_input = await asyncio.to_thread(console.input, "")

                if not user_input.strip():
                    continue

                if user_input.lower() in ("exit", "quit"):
                    break

                if user_input.lower() == "clear":
                    chat.reset()
                    ui.text(Color.USER, "Cleared")
                    ui.delimiter(Color.USER)
                    continue

                ui.delimiter(Color.USER)
                await chat.turn(user_input)

        finally:
            await llm.aclose()



async def main() -> None:
    try:
        config = AppConfig()
    except ValueError as e:
        print(f"Configuration error: {e}")
        return
    
    console = Console()
    stop = asyncio.Event()

    def handle_signal() -> None:
        stop.set()
        console.print("\n[yellow]Shutting down...[/yellow]")

    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT)):
        try:
            signal.signal(sig, lambda *_: handle_signal())
        except Exception:
            pass

    repl_task = asyncio.create_task(run_repl(config, console))
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