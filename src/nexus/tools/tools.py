from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

load_dotenv()

# ---------------------------------------------------------------------
# TOOLS
# ---------------------------------------------------------------------

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode


_tools = None
_tool_node = None

async def get_app_resources():
    global _tools, _tool_node
    if _tools is None:
        # One-time initialization of the connection
        mcp_client = MultiServerMCPClient({
            "TicketService": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            }
        })
        _tools = await mcp_client.get_tools()
        _tool_node = ToolNode(_tools)
    return _tool_node, _tools

async def get_tools_node():
    mcp_client = MultiServerMCPClient({
        "TicketService": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        }
    })
    # Fetch the actual tool objects from the MCP server
    mcp_tools = await mcp_client.get_tools()

    # Return the executable node AND the tool definitions
    return ToolNode(mcp_tools), mcp_tools