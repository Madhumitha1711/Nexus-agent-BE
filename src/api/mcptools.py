from typing import Optional, Dict, Any
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP
mcp = FastMCP("TicketService")

# In-memory mock database
TICKET_DB: Dict[str, Dict[str, Any]] = {
    "TKT-101": {
        "ticket_id": "TKT-101",
        "subject": "Login issue",
        "description": "User cannot log in with valid credentials",
        "status": "Open",
        "priority": "High",
    },
    "TKT-102": {
        "ticket_id": "TKT-102",
        "subject": "Payment failed",
        "description": "Transaction fails during checkout",
        "status": "Resolved",
        "priority": "Medium",
    },
}


@mcp.tool()
def get_ticket(ticket_id: str) -> Dict[str, Any]:
    """Retrieve a ticket by ID."""
    ticket = TICKET_DB.get(ticket_id)

    if not ticket:
        return {
            "success": False,
            "error": "Ticket not found",
            "ticket_id": ticket_id,
        }

    return {
        "success": True,
        "ticket": ticket,
    }


@mcp.tool()
def create_ticket(
    subject: str,
    description: str,
    priority: str = "Medium",
) -> Dict[str, Any]:
    """Create a new support ticket."""
    new_id = f"TKT-{100 + len(TICKET_DB) + 1}"

    ticket = {
        "ticket_id": new_id,
        "subject": subject,
        "description": description,
        "status": "Open",
        "priority": priority,
    }

    TICKET_DB[new_id] = ticket

    return {
        "success": True,
        "ticket": ticket,
    }


@mcp.tool()
def update_ticket(
    ticket_id: str,
    new_status: Optional[str] = None,
    comment: Optional[str] = None,
) -> Dict[str, Any]:
    """Update ticket status or add a comment."""
    ticket = TICKET_DB.get(ticket_id)

    if not ticket:
        return {
            "success": False,
            "error": "Ticket not found",
            "ticket_id": ticket_id,
        }

    if new_status:
        ticket["status"] = new_status

    # Note: comments are acknowledged but not stored in this mock DB
    return {
        "success": True,
        "ticket_id": ticket_id,
        "updated_status": ticket.get("status"),
        "comment": comment,
    }


if __name__ == "__main__":
    # Start MCP server using streamable HTTP transport
    mcp.run(transport="streamable-http")
