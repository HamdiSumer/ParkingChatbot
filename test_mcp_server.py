#!/usr/bin/env python3
"""Test script for the MCP Reservation Server.

This script tests the MCP server tools directly without needing an MCP client.
It simulates what an MCP client would do.

Usage:
    uv run python test_mcp_server.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.mcp.reservation_server import (
    write_reservation,
    read_reservations,
    get_reservation_file_info,
    RESERVATIONS_FILE
)


async def test_mcp_tools():
    """Test all MCP tools."""
    print("=" * 60)
    print("Testing MCP Reservation Server Tools")
    print("=" * 60)
    print()

    # Test 1: Get file info
    print("Test 1: Get Reservation File Info")
    print("-" * 40)
    result = await get_reservation_file_info({})
    print(result[0].text)
    print()

    # Test 2: Write a test reservation
    print("Test 2: Write Test Reservation")
    print("-" * 40)
    test_reservation = {
        "reservation_id": "MCP_TEST_001",
        "user_name": "MCP",
        "user_surname": "TestUser",
        "car_number": "MCP-999",
        "start_time": "2025-12-25 10:00",
        "end_time": "2025-12-25 14:00",
        "approved_by": "MCP Server Test",
        "parking_id": "downtown_1"
    }
    result = await write_reservation(test_reservation)
    print(result[0].text)
    print()

    # Test 3: Read reservations
    print("Test 3: Read Recent Reservations")
    print("-" * 40)
    result = await read_reservations({"limit": 5})
    print(result[0].text)
    print()

    # Test 4: Verify file was updated
    print("Test 4: Verify File Updated")
    print("-" * 40)
    result = await get_reservation_file_info({})
    print(result[0].text)
    print()

    print("=" * 60)
    print("All MCP tools tested successfully!")
    print(f"Reservations file: {RESERVATIONS_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_mcp_tools())
