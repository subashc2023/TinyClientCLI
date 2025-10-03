uv run ./main.py


Sample mcp_config.json here
{
    "mcpServers": {
        "filesystem": {
            "command": "npx",
            "args": [
              "-y",
              "@modelcontextprotocol/server-filesystem",
              "./"
            ]
        }
    }
}