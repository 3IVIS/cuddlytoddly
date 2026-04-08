# tools/mcp_adapter.py
#
# Bridges MCP servers into the existing ToolRegistry so the LLMExecutor
# can call any MCP tool without knowing anything about MCP.
#
# Usage
# -----
# from tools.mcp_adapter import MCPAdapter
#
# # Filesystem server (reads/writes local files)
# adapter = MCPAdapter.from_stdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
# registry = adapter.build_registry()
#
# # Sequential thinking server
# adapter = MCPAdapter.from_stdio("npx", ["-y", "@modelcontextprotocol/server-sequential-thinking"])
# registry = adapter.build_registry()
#
# # Multiple servers merged into one registry
# registry = MCPAdapter.merged_registry([
#     MCPAdapter.from_stdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]),
#     MCPAdapter.from_stdio("npx", ["-y", "@modelcontextprotocol/server-brave-search"]),
# ])
#
# Then pass registry to LLMExecutor:
#   executor = LLMExecutor(llm_client=shared_llm, tool_registry=registry, max_turns=5)
#
# Dependencies
# ------------
#   pip install mcp
#
# Popular ready-made MCP servers (all via npx, no install needed):
#   @modelcontextprotocol/server-filesystem      read/write local files
#   @modelcontextprotocol/server-memory          persistent key-value memory
#   @modelcontextprotocol/server-brave-search    web search (needs BRAVE_API_KEY)
#   @modelcontextprotocol/server-github          GitHub API
#   @modelcontextprotocol/server-postgres        PostgreSQL queries
#   @modelcontextprotocol/server-sqlite          SQLite queries
#   @modelcontextprotocol/server-sequential-thinking  chain-of-thought reasoning


import asyncio
import json
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from cuddlytoddly.infra.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Inline ToolRegistry (mirrors engine/tools.py to avoid a circular import)
# ---------------------------------------------------------------------------


class Tool:
    def __init__(self, name: str, description: str, input_schema: dict, fn):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self._fn = fn

    def run(self, input_data: dict) -> Any:
        return self._fn(input_data)


class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool
        logger.info("[TOOLS] Registered tool: %s", tool.name)

    def execute(self, name: str, input_data: dict) -> Any:
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self.tools[name].run(input_data)


# ---------------------------------------------------------------------------
# MCP Adapter
# ---------------------------------------------------------------------------


class MCPAdapter:
    """
    Connects to an MCP server, discovers its tools, and exposes them
    as a populated ToolRegistry.

    Parameters
    ----------
    server_params : mcp.StdioServerParameters
        How to launch the MCP server process.
    """

    def __init__(self, server_params):
        self._params = server_params

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_stdio(
        cls, command: str, args: list[str], env: dict | None = None
    ) -> "MCPAdapter":
        """
        Create an adapter for a stdio MCP server (the most common kind).

        Parameters
        ----------
        command : str        e.g. "npx" or "python"
        args    : list[str]  e.g. ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        env     : dict | None  extra environment variables (e.g. {"BRAVE_API_KEY": "..."})
        """
        try:
            from mcp import StdioServerParameters
        except ImportError as e:
            raise ImportError("mcp is not installed. Run: pip install mcp") from e

        params = StdioServerParameters(command=command, args=args, env=env)
        return cls(params)

    # ── Registry builder ──────────────────────────────────────────────────────

    def build_registry(self) -> ToolRegistry:
        """
        Launch the MCP server, list its tools, and return a ToolRegistry.

        Each MCP tool becomes a callable Tool that synchronously calls back
        into the server via a fresh async session (one call per invocation).
        """
        try:
            import mcp  # noqa: F401
        except ImportError as e:
            raise ImportError("mcp is not installed. Run: pip install mcp") from e

        # Discover available tools synchronously
        tool_defs = asyncio.run(self._list_tools())
        registry = ToolRegistry()

        for t in tool_defs:
            # Capture t.name in the closure correctly
            tool_name = t.name

            def make_fn(name):
                def call_tool(input_data: dict) -> str:
                    return asyncio.run(self._call_tool(name, input_data))

                return call_tool

            registry.register(
                Tool(
                    name=tool_name,
                    description=t.description or "",
                    input_schema=t.inputSchema if hasattr(t, "inputSchema") else {},
                    fn=make_fn(tool_name),
                )
            )

        logger.info(
            "[MCP] Registry built with %d tool(s) from server", len(registry.tools)
        )
        return registry

    # ── Async helpers ─────────────────────────────────────────────────────────

    async def _list_tools(self) -> list:

        async with stdio_client(self._params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await session.list_tools()
                return response.tools

    async def _call_tool(self, name: str, arguments: dict) -> str:

        async with stdio_client(self._params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)

                # MCP returns a list of content blocks; join text ones
                parts = []
                for block in result.content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                    else:
                        parts.append(json.dumps(block.__dict__))

                return "\n".join(parts)

    # ── Multi-server merge ────────────────────────────────────────────────────

    @staticmethod
    def merged_registry(adapters: list["MCPAdapter"]) -> ToolRegistry:
        """
        Build a single ToolRegistry from multiple MCP servers.
        Later adapters win on name collision.
        """
        merged = ToolRegistry()
        for adapter in adapters:
            sub = adapter.build_registry()
            for tool in sub.tools.values():
                merged.register(tool)
        return merged
