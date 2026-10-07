import sys

import pytest

from mcp import Client, StdioServerParameters


@pytest.mark.anyio
async def test_mcp_server_works_over_stdio():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp.server"],
    )

    async with Client(server_params) as client:
        result = await client.list_tools()

    tool_names = [
        tool.name
        for tool in result.tools
    ]

    assert "health_check" in tool_names
    assert "research" in tool_names