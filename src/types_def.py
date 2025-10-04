from typing import List, Literal, TypedDict


Role = Literal["system", "user", "assistant", "tool"]


class ToolFunction(TypedDict, total=False):
    name: str
    arguments: str


class ToolCall(TypedDict, total=False):
    id: str
    type: str
    function: ToolFunction


class Message(TypedDict, total=False):
    role: Role
    content: str
    tool_calls: List[ToolCall]
    tool_call_id: str


class ToolResult(TypedDict):
    tool_call_id: str
    content: str
    is_error: bool