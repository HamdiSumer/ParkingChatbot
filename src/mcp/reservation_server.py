"""MCP Server for Parking Reservation Management.

This server exposes tools for:
- Writing confirmed reservations to file
- Reading reservation history
- Checking reservation file status

Security Features:
- API key authentication (optional)
- Rate limiting (requests per minute)
- Input sanitization
- File locking for concurrent access
- Access logging/audit trail

Designed to work with:
- Claude Desktop / Claude Code (native MCP support)
- Ollama via MCP-Ollama bridge (https://github.com/Sethuram2003/MCP-ollama_server)

Usage:
    # Run the server
    uv run python -m src.mcp.reservation_server

    # Or via MCP CLI
    mcp run src/mcp/reservation_server.py
"""

import os
import sys
import fcntl
import hashlib
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Initialize MCP server
mcp_server = Server("parking-reservations")

# Reservation file path
RESERVATIONS_DIR = project_root / "data" / "confirmed_reservations"
RESERVATIONS_FILE = RESERVATIONS_DIR / "confirmed_reservations.txt"
ACCESS_LOG_FILE = RESERVATIONS_DIR / "access_log.txt"

# Security configuration
MCP_API_KEY = os.environ.get("MCP_API_KEY", None)  # Optional API key
REQUIRE_AUTH = os.environ.get("MCP_REQUIRE_AUTH", "false").lower() == "true"
RATE_LIMIT_REQUESTS = int(os.environ.get("MCP_RATE_LIMIT", "60"))  # per minute
RATE_LIMIT_WINDOW = 60  # seconds


# ==================== SECURITY ====================

class RateLimiter:
    """Simple rate limiter for MCP requests."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)

    def is_allowed(self, client_id: str = "default") -> bool:
        """Check if request is allowed."""
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests
        self.requests[client_id] = [
            t for t in self.requests[client_id] if t > window_start
        ]

        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False

        self.requests[client_id].append(now)
        return True

    def get_remaining(self, client_id: str = "default") -> int:
        """Get remaining requests in window."""
        now = time.time()
        window_start = now - self.window_seconds
        current = len([t for t in self.requests[client_id] if t > window_start])
        return max(0, self.max_requests - current)


# Global rate limiter
rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW)


def verify_api_key(provided_key: Optional[str]) -> bool:
    """Verify API key if authentication is required."""
    if not REQUIRE_AUTH:
        return True
    if not MCP_API_KEY:
        return True  # No key configured, allow all
    if not provided_key:
        return False
    # Constant-time comparison to prevent timing attacks
    return hashlib.sha256(provided_key.encode()).hexdigest() == hashlib.sha256(MCP_API_KEY.encode()).hexdigest()


def log_access(action: str, details: str, success: bool = True):
    """Log access to the audit trail."""
    try:
        ensure_file_exists()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "SUCCESS" if success else "FAILED"
        log_line = f"[{timestamp}] [{status}] {action}: {details}\n"

        with open(ACCESS_LOG_FILE, 'a') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(log_line)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass  # Don't fail on logging errors


def check_security(action: str, api_key: Optional[str] = None) -> Optional[str]:
    """Check security constraints. Returns error message or None if OK."""
    # Rate limiting
    if not rate_limiter.is_allowed():
        log_access(action, "Rate limit exceeded", success=False)
        return f"Rate limit exceeded. Max {RATE_LIMIT_REQUESTS} requests per minute. Try again later."

    # API key verification
    if not verify_api_key(api_key):
        log_access(action, "Invalid API key", success=False)
        return "Authentication failed. Invalid or missing API key."

    return None


# ==================== FILE OPERATIONS ====================

def ensure_file_exists():
    """Ensure the reservations directory and file exist."""
    RESERVATIONS_DIR.mkdir(parents=True, exist_ok=True)
    if not RESERVATIONS_FILE.exists():
        header = create_header()
        with open(RESERVATIONS_FILE, 'w') as f:
            f.write(header)


def create_header() -> str:
    """Create the file header."""
    separator = "=" * 120
    header_line = f"{'Name':<25} | {'Car Number':<15} | {'Reservation Period':<40} | {'Approval Time':<20} | {'Admin':<15}"
    return f"{separator}\nCONFIRMED PARKING RESERVATIONS\n{separator}\n{header_line}\n{separator}\n"


def sanitize_input(value: str, max_length: int = 50) -> str:
    """Sanitize input to prevent injection attacks."""
    if not value:
        return "N/A"
    value = str(value).strip()
    # Remove dangerous characters
    value = value.replace('|', '-')  # Delimiter
    value = value.replace('\n', ' ').replace('\r', ' ')  # Newlines
    value = value.replace('\x00', '')  # Null bytes
    value = value.replace('..', '')  # Path traversal
    # Truncate
    if len(value) > max_length:
        value = value[:max_length]
    return value


# ==================== MCP TOOLS ====================

@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="write_reservation",
            description="Write a confirmed parking reservation to the storage file. Call this when an admin approves a reservation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "reservation_id": {
                        "type": "string",
                        "description": "Unique reservation ID (e.g., RES_ABC123)"
                    },
                    "user_name": {
                        "type": "string",
                        "description": "User's first name"
                    },
                    "user_surname": {
                        "type": "string",
                        "description": "User's last name"
                    },
                    "car_number": {
                        "type": "string",
                        "description": "Car registration/license plate number"
                    },
                    "start_time": {
                        "type": "string",
                        "description": "Reservation start time (ISO format or YYYY-MM-DD HH:MM)"
                    },
                    "end_time": {
                        "type": "string",
                        "description": "Reservation end time (ISO format or YYYY-MM-DD HH:MM)"
                    },
                    "approved_by": {
                        "type": "string",
                        "description": "Name of the admin who approved the reservation"
                    },
                    "parking_id": {
                        "type": "string",
                        "description": "Parking location ID (e.g., downtown_1)"
                    }
                },
                "required": ["reservation_id", "user_name", "user_surname", "car_number", "start_time", "end_time", "approved_by"]
            }
        ),
        Tool(
            name="read_reservations",
            description="Read all confirmed reservations from the storage file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of reservations to return (default: 10)"
                    }
                }
            }
        ),
        Tool(
            name="get_reservation_file_info",
            description="Get information about the reservation file (path, size, last modified).",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""

    if name == "write_reservation":
        return await write_reservation(arguments)
    elif name == "read_reservations":
        return await read_reservations(arguments)
    elif name == "get_reservation_file_info":
        return await get_reservation_file_info(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def write_reservation(args: dict) -> list[TextContent]:
    """Write a confirmed reservation to file."""
    # Security check
    api_key = args.get("api_key")
    security_error = check_security("write_reservation", api_key)
    if security_error:
        return [TextContent(type="text", text=f"SECURITY ERROR: {security_error}")]

    try:
        ensure_file_exists()

        # Extract and sanitize arguments
        reservation_id = sanitize_input(args.get("reservation_id", ""), 20)
        user_name = sanitize_input(args.get("user_name", ""), 15)
        user_surname = sanitize_input(args.get("user_surname", ""), 15)
        car_number = sanitize_input(args.get("car_number", ""), 15)
        start_time = args.get("start_time", "")
        end_time = args.get("end_time", "")
        approved_by = sanitize_input(args.get("approved_by", "MCP"), 15)
        parking_id = sanitize_input(args.get("parking_id", "N/A"), 15)

        # Format times
        def format_time(t):
            if isinstance(t, str):
                try:
                    dt = datetime.fromisoformat(t.replace('Z', '+00:00'))
                    return dt.strftime("%Y-%m-%d %H:%M")
                except:
                    return t
            return str(t)

        start_formatted = format_time(start_time)
        end_formatted = format_time(end_time)
        approval_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Create formatted line
        full_name = f"{user_name} {user_surname}"[:25]
        period = f"{start_formatted} to {end_formatted}"
        line = f"{full_name:<25} | {car_number:<15} | {period:<40} | {approval_time:<20} | {approved_by:<15}\n"

        # Write to file with file locking for concurrent access
        with open(RESERVATIONS_FILE, 'a') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        # Log successful access
        log_access("write_reservation", f"ID={reservation_id}, User={full_name}, Car={car_number}")

        return [TextContent(
            type="text",
            text=f"SUCCESS: Reservation {reservation_id} written to file.\n"
                 f"Name: {full_name}\n"
                 f"Car: {car_number}\n"
                 f"Period: {period}\n"
                 f"Approved by: {approved_by}\n"
                 f"File: {RESERVATIONS_FILE}"
        )]

    except Exception as e:
        log_access("write_reservation", f"Error: {str(e)}", success=False)
        return [TextContent(
            type="text",
            text=f"ERROR: Failed to write reservation: {str(e)}"
        )]


async def read_reservations(args: dict) -> list[TextContent]:
    """Read reservations from file."""
    # Security check
    api_key = args.get("api_key")
    security_error = check_security("read_reservations", api_key)
    if security_error:
        return [TextContent(type="text", text=f"SECURITY ERROR: {security_error}")]

    try:
        ensure_file_exists()

        limit = args.get("limit", 10)

        # Read with file locking for consistency
        with open(RESERVATIONS_FILE, 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
            try:
                lines = f.readlines()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        # Skip header (first 5 lines)
        data_lines = [line.strip() for line in lines[5:] if line.strip()]

        if not data_lines:
            log_access("read_reservations", "No reservations found")
            return [TextContent(type="text", text="No confirmed reservations found.")]

        # Apply limit
        if limit and limit > 0:
            data_lines = data_lines[-limit:]

        log_access("read_reservations", f"Retrieved {len(data_lines)} reservation(s)")

        result = f"Found {len(data_lines)} reservation(s):\n\n"
        result += "\n".join(data_lines)

        return [TextContent(type="text", text=result)]

    except Exception as e:
        log_access("read_reservations", f"Error: {str(e)}", success=False)
        return [TextContent(
            type="text",
            text=f"ERROR: Failed to read reservations: {str(e)}"
        )]


async def get_reservation_file_info(args: dict = None) -> list[TextContent]:
    """Get file information."""
    args = args or {}

    # Security check
    api_key = args.get("api_key")
    security_error = check_security("get_reservation_file_info", api_key)
    if security_error:
        return [TextContent(type="text", text=f"SECURITY ERROR: {security_error}")]

    try:
        ensure_file_exists()

        stat = RESERVATIONS_FILE.stat()
        size_kb = stat.st_size / 1024
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

        # Count reservations with file locking
        with open(RESERVATIONS_FILE, 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                lines = f.readlines()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        reservation_count = len([l for l in lines[5:] if l.strip()])

        log_access("get_reservation_file_info", f"File size: {size_kb:.2f}KB, Count: {reservation_count}")

        return [TextContent(
            type="text",
            text=f"Reservation File Info:\n"
                 f"  Path: {RESERVATIONS_FILE}\n"
                 f"  Size: {size_kb:.2f} KB\n"
                 f"  Last Modified: {modified}\n"
                 f"  Total Reservations: {reservation_count}"
        )]

    except Exception as e:
        log_access("get_reservation_file_info", f"Error: {str(e)}", success=False)
        return [TextContent(
            type="text",
            text=f"ERROR: Failed to get file info: {str(e)}"
        )]


# ==================== MAIN ====================

async def main():
    """Run the MCP server."""
    print("Starting Parking Reservation MCP Server...", file=sys.stderr)
    print(f"Reservations file: {RESERVATIONS_FILE}", file=sys.stderr)

    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
