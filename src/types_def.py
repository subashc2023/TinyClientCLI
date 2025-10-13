from typing import List, Literal, TypedDict, Optional
from dataclasses import dataclass, field

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


@dataclass
class UsageStats:
    """Track token usage and cost across conversation."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Model pricing per 1M tokens (update as needed)
    MODEL_PRICING = {
        "gpt-4o": {"prompt": 2.50, "completion": 10.00},
        "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
        "gpt-4-turbo": {"prompt": 10.00, "completion": 30.00},
        "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
    }
    
    def update(self, usage: Optional[dict]) -> None:
        """Update stats from API response usage object (tolerates None)."""
        if not usage:
            return
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.total_tokens += usage.get("total_tokens", 0)
    
    def cost_estimate(self, model: str) -> float:
        """Estimate cost in USD for given model."""
        pricing = self.MODEL_PRICING.get(model, {"prompt": 0, "completion": 0})
        prompt_cost = (self.prompt_tokens / 1_000_000) * pricing["prompt"]
        completion_cost = (self.completion_tokens / 1_000_000) * pricing["completion"]
        return prompt_cost + completion_cost
    
    def format_summary(self, model: str) -> str:
        """Format usage summary as string."""
        cost = self.cost_estimate(model)
        return (
            f"Tokens: {self.total_tokens:,} "
            f"(prompt: {self.prompt_tokens:,}, completion: {self.completion_tokens:,}) "
            f"| Cost: ${cost:.4f}"
        )


@dataclass
class MCPResource:
    """Represents an MCP resource."""
    server_name: str
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None