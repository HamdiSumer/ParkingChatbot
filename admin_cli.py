#!/usr/bin/env python3
"""Admin CLI for managing parking reservations.

This CLI uses the LangChain Admin Agent which communicates via REST API
with fallback to direct database access when API is unavailable.
"""
import sys
import httpx
from datetime import datetime
from typing import Optional

from src.agents.admin_agent import AdminAgent
from src.config import get_config

# Initialize config to ensure environment is loaded
get_config()

# API configuration
API_BASE_URL = "http://localhost:8001"


def print_header():
    """Print CLI header."""
    print("\n" + "=" * 60)
    print("       Admin Agent - Parking Reservation System")
    print("=" * 60)


def print_help():
    """Print available commands."""
    print("""
Commands:
  list              - Show all pending reservations
  approve <id>      - Approve a reservation
  reject <id>       - Reject a reservation (will ask for reason)
  status <id>       - Check reservation status/details
  history           - Show recently reviewed reservations
  notifications     - Check for new notifications
  agent <query>     - Run a query through the Admin Agent (AI)
  api-status        - Check if REST API is running
  help              - Show this help message
  quit              - Exit the admin CLI
""")


def format_datetime(dt) -> str:
    """Format datetime for display."""
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d %H:%M")
    return str(dt) if dt else "N/A"


def check_api_status() -> bool:
    """Check if the REST API is running."""
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"\n[OK] API is running")
                print(f"  Service: {data.get('service', 'unknown')}")
                print(f"  Status: {data.get('status', 'unknown')}")
                print(f"  Timestamp: {data.get('timestamp', 'unknown')}")
                return True
    except httpx.ConnectError:
        print(f"\n[WARN] API not available at {API_BASE_URL}")
        print("  Using direct database access as fallback.")
        return False
    except Exception as e:
        print(f"\n[ERROR] API check failed: {e}")
        return False
    return False


def list_pending_api() -> Optional[list]:
    """List pending reservations via API."""
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{API_BASE_URL}/api/reservations/pending/list")
            if response.status_code == 200:
                data = response.json()
                return data.get("reservations", [])
    except httpx.ConnectError:
        return None
    return None


def get_reservation_api(res_id: str) -> Optional[dict]:
    """Get reservation details via API."""
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{API_BASE_URL}/api/reservations/{res_id}")
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None
    except httpx.ConnectError:
        return None
    return None


def admin_decision_api(res_id: str, decision: str, admin_name: str,
                       reason: Optional[str] = None, notes: Optional[str] = None) -> dict:
    """Submit admin decision via API."""
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{API_BASE_URL}/api/admin/decision",
                json={
                    "reservation_id": res_id,
                    "decision": decision,
                    "admin_name": admin_name,
                    "reason": reason,
                    "notes": notes,
                }
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "message": f"API error: {response.status_code}"}
    except httpx.ConnectError:
        return {"success": False, "message": "API not available"}
    except Exception as e:
        return {"success": False, "message": str(e)}


def get_notifications_api() -> list:
    """Get pending notifications from API."""
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{API_BASE_URL}/api/notifications/pending")
            if response.status_code == 200:
                return response.json()
    except:
        pass
    return []


def list_pending(admin_agent: AdminAgent):
    """List all pending reservations."""
    # Try API first
    reservations = list_pending_api()

    if reservations is None:
        # Fallback to agent
        result = admin_agent._list_pending_reservations("")
        print(f"\n{result}")
        return

    if not reservations:
        print("\nNo pending reservations.")
        return

    print("\nPending Reservations:")
    print("-" * 80)
    print(f"{'ID':<14} | {'Name':<18} | {'Parking':<12} | {'Time':<20}")
    print("-" * 80)

    for res in reservations:
        name = f"{res['user_name']} {res['user_surname']}"
        time_str = format_datetime(res.get('start_time', 'N/A'))
        parking = res.get('parking_name', res.get('parking_id', 'N/A'))
        print(f"{res['id']:<14} | {name:<18} | {parking:<12} | {time_str:<20}")

    print("-" * 80)
    print(f"Total: {len(reservations)} pending reservation(s)")


def approve_reservation(admin_agent: AdminAgent, res_id: str):
    """Approve a reservation."""
    # Get reservation details (try API first)
    reservation = get_reservation_api(res_id)

    if reservation is None:
        # Fallback to agent
        status_text = admin_agent._check_reservation_status(res_id)
        if "not found" in status_text.lower():
            print(f"\nReservation {res_id} not found.")
            return
        print(f"\n{status_text}")
        # If fallback, still continue with approval
        reservation = {"id": res_id, "status": "pending"}

    if reservation.get("status") != "pending":
        print(f"\nReservation {res_id} is not pending (status: {reservation.get('status', 'unknown')})")
        return

    # Show reservation details
    print(f"\nReservation Details:")
    print(f"  ID: {reservation.get('id', res_id)}")
    if reservation.get('user_name'):
        print(f"  Name: {reservation['user_name']} {reservation.get('user_surname', '')}")
    if reservation.get('car_number'):
        print(f"  Car: {reservation['car_number']}")
    if reservation.get('parking_name') or reservation.get('parking_id'):
        print(f"  Parking: {reservation.get('parking_name', reservation.get('parking_id', 'N/A'))}")
    if reservation.get('start_time'):
        print(f"  Time: {format_datetime(reservation['start_time'])} - {format_datetime(reservation.get('end_time'))}")

    # Get admin name
    admin_name = input("\nEnter your admin name: ").strip()
    if not admin_name:
        print("Admin name is required. Approval cancelled.")
        return

    # Get optional notes
    notes = input("Notes (optional, press Enter to skip): ").strip()

    # Confirm
    confirm = input(f"\nApprove reservation {res_id}? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Approval cancelled.")
        return

    # Approve via API or agent
    result = admin_decision_api(res_id, "approve", admin_name, notes=notes if notes else None)

    if not result.get("success") and "API not available" in result.get("message", ""):
        # Fallback to agent
        result_text = admin_agent._approve_reservation(res_id, admin_name, notes if notes else None)
        if "SUCCESS" in result_text:
            print(f"\n[OK] {result_text}")
        else:
            print(f"\n[ERROR] {result_text}")
        return

    if result.get("success"):
        print(f"\n[OK] {result.get('message', 'Reservation approved')}")
    else:
        print(f"\n[ERROR] {result.get('message', 'Failed to approve')}")


def reject_reservation(admin_agent: AdminAgent, res_id: str):
    """Reject a reservation."""
    # Get reservation details (try API first)
    reservation = get_reservation_api(res_id)

    if reservation is None:
        # Fallback to agent
        status_text = admin_agent._check_reservation_status(res_id)
        if "not found" in status_text.lower():
            print(f"\nReservation {res_id} not found.")
            return
        print(f"\n{status_text}")
        reservation = {"id": res_id, "status": "pending"}

    if reservation.get("status") != "pending":
        print(f"\nReservation {res_id} is not pending (status: {reservation.get('status', 'unknown')})")
        return

    # Show reservation details
    print(f"\nReservation Details:")
    print(f"  ID: {reservation.get('id', res_id)}")
    if reservation.get('user_name'):
        print(f"  Name: {reservation['user_name']} {reservation.get('user_surname', '')}")
    if reservation.get('car_number'):
        print(f"  Car: {reservation['car_number']}")
    if reservation.get('parking_name') or reservation.get('parking_id'):
        print(f"  Parking: {reservation.get('parking_name', reservation.get('parking_id', 'N/A'))}")

    # Get admin name
    admin_name = input("\nEnter your admin name: ").strip()
    if not admin_name:
        print("Admin name is required. Rejection cancelled.")
        return

    # Get rejection reason (required)
    reason = input("Reason for rejection (required): ").strip()
    if not reason:
        print("Rejection reason is required. Rejection cancelled.")
        return

    # Confirm
    confirm = input(f"\nReject reservation {res_id}? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Rejection cancelled.")
        return

    # Reject via API or agent
    result = admin_decision_api(res_id, "reject", admin_name, reason=reason)

    if not result.get("success") and "API not available" in result.get("message", ""):
        # Fallback to agent
        result_text = admin_agent._reject_reservation(res_id, admin_name, reason)
        if "SUCCESS" in result_text:
            print(f"\n[OK] {result_text}")
        else:
            print(f"\n[ERROR] {result_text}")
        return

    if result.get("success"):
        print(f"\n[OK] {result.get('message', 'Reservation rejected')}")
    else:
        print(f"\n[ERROR] {result.get('message', 'Failed to reject')}")


def show_status(admin_agent: AdminAgent, res_id: str):
    """Show reservation status and details."""
    # Try API first
    reservation = get_reservation_api(res_id)

    if reservation is None:
        # Fallback to agent
        result = admin_agent._check_reservation_status(res_id)
        print(f"\n{result}")
        return

    if not reservation:
        print(f"\nReservation {res_id} not found.")
        return

    status_display = {
        "pending": "PENDING (awaiting review)",
        "confirmed": "APPROVED",
        "rejected": "REJECTED",
        "completed": "COMPLETED",
        "cancelled": "CANCELLED",
    }

    print(f"\nReservation Status: {res_id}")
    print("-" * 50)
    print(f"  Status: {status_display.get(reservation['status'], reservation['status'])}")
    print(f"  Name: {reservation.get('user_name', 'N/A')} {reservation.get('user_surname', '')}")
    print(f"  Car: {reservation.get('car_number', 'N/A')}")
    print(f"  Parking: {reservation.get('parking_name', reservation.get('parking_id', 'N/A'))}")
    print(f"  Time: {format_datetime(reservation.get('start_time'))} - {format_datetime(reservation.get('end_time'))}")
    print(f"  Created: {format_datetime(reservation.get('created_at'))}")

    if reservation.get('status') == 'confirmed':
        print(f"  Approved by: {reservation.get('approved_by_admin', 'N/A')}")
        print(f"  Reviewed at: {format_datetime(reservation.get('reviewed_at'))}")
        if reservation.get('admin_notes'):
            print(f"  Notes: {reservation['admin_notes']}")

    elif reservation.get('status') == 'rejected':
        print(f"  Rejected by: {reservation.get('approved_by_admin', 'N/A')}")
        print(f"  Reason: {reservation.get('rejection_reason', 'N/A')}")
        print(f"  Reviewed at: {format_datetime(reservation.get('reviewed_at'))}")


def show_history(admin_agent: AdminAgent):
    """Show recently reviewed reservations."""
    # Use admin agent's service layer directly for history
    reservations = admin_agent.admin_service.get_reviewed_history(20)

    if not reservations:
        print("\nNo reviewed reservations yet.")
        return

    print("\nRecently Reviewed Reservations:")
    print("-" * 90)
    print(f"{'ID':<14} | {'Name':<18} | {'Status':<10} | {'Reviewed At':<16} | {'By':<10}")
    print("-" * 90)

    for res in reservations:
        name = f"{res.get('user_name', '')} {res.get('user_surname', '')}"
        status = "APPROVED" if res.get('status') == 'confirmed' else "REJECTED"
        reviewed = format_datetime(res.get('reviewed_at'))
        admin = res.get('approved_by_admin') or "N/A"
        print(f"{res['id']:<14} | {name:<18} | {status:<10} | {reviewed:<16} | {admin:<10}")

    print("-" * 90)


def show_notifications():
    """Show pending notifications."""
    notifications = get_notifications_api()

    if not notifications:
        print("\nNo pending notifications.")
        return

    print(f"\nPending Notifications ({len(notifications)}):")
    print("-" * 60)

    for notif in notifications:
        priority = notif.get('priority', 'normal').upper()
        priority_marker = "[!]" if priority == "HIGH" else "[ ]"
        print(f"{priority_marker} {notif.get('type', 'unknown')}")
        print(f"    {notif.get('message', 'No message')}")
        print(f"    Time: {notif.get('timestamp', 'unknown')}")
        print()


def run_agent_query(admin_agent: AdminAgent, query: str):
    """Run a query through the Admin Agent AI."""
    print("\nProcessing query with Admin Agent...")
    result = admin_agent.run(query)
    print(f"\nAgent Response:\n{result}")


def main():
    """Main CLI loop."""
    print_header()

    # Check API status on startup
    api_available = check_api_status()
    if not api_available:
        print("\n  Initializing Admin Agent with direct database access...")

    print_help()

    # Initialize Admin Agent (will use API if available, fallback to DB)
    admin_agent = AdminAgent(use_api=api_available)

    try:
        while True:
            try:
                # Check for new notifications periodically (show indicator)
                notifications = get_notifications_api()
                notif_indicator = f" [{len(notifications)} new]" if notifications else ""

                user_input = input(f"\nadmin{notif_indicator}> ").strip()

                if not user_input:
                    continue

                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                if command == "quit" or command == "exit":
                    print("\nGoodbye!")
                    break

                elif command == "help":
                    print_help()

                elif command == "list":
                    list_pending(admin_agent)

                elif command == "approve":
                    if not args:
                        print("Usage: approve <reservation_id>")
                    else:
                        approve_reservation(admin_agent, args.strip().upper())

                elif command == "reject":
                    if not args:
                        print("Usage: reject <reservation_id>")
                    else:
                        reject_reservation(admin_agent, args.strip().upper())

                elif command == "status":
                    if not args:
                        print("Usage: status <reservation_id>")
                    else:
                        show_status(admin_agent, args.strip().upper())

                elif command == "history":
                    show_history(admin_agent)

                elif command == "notifications":
                    show_notifications()

                elif command == "agent":
                    if not args:
                        print("Usage: agent <query>")
                        print("Example: agent list all pending reservations")
                    else:
                        run_agent_query(admin_agent, args)

                elif command == "api-status":
                    check_api_status()

                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands.")

            except KeyboardInterrupt:
                print("\n\nUse 'quit' to exit.")

    finally:
        admin_agent.close()


if __name__ == "__main__":
    main()
