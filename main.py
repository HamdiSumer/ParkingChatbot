#!/usr/bin/env python3
"""Entry point for the Parking Chatbot with Human-in-the-Loop (HITL).

This chatbot PAUSES and WAITS for admin approval when making reservations.

Usage:
1. Start the admin server (in another terminal):
   uv run uvicorn src.api.server:app --port 8001

2. Run this chatbot:
   uv run python main.py

3. Make a reservation - the bot will PAUSE after submission
4. Go to http://localhost:8001/dashboard/ and click Approve/Reject
5. The bot will automatically show the result!

For the old CLI without HITL, run: uv run python -m src.cli
"""
import os
import sys
import time
import threading
import warnings
from datetime import datetime

# Suppress verbose logging and warnings for clean user experience
os.environ['LOG_LEVEL'] = 'WARNING'
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import get_config
from src.agents.hitl_workflow import HITLWorkflow
from src.utils.logging import logger

# Get config
config = get_config()


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 70)
    print("   PARKING CHATBOT - Human-in-the-Loop (HITL) Mode")
    print("=" * 70)
    print("   This bot PAUSES when you make a reservation and WAITS for admin!")
    print("")
    print("   1. Start admin server: uv run uvicorn src.api.server:app --port 8001")
    print("   2. Dashboard: http://localhost:8001/dashboard/")
    print("   3. Make a reservation here, then approve/reject on dashboard")
    print("=" * 70)
    print("")


def check_for_admin_response(workflow, reservation_id, thread_id, stop_event, shared_state):
    """Background thread to check if admin has responded."""
    while not stop_event.is_set():
        # Check if admin has responded (status changed to 'admin_responded')
        admin_response = workflow.thread_store.check_admin_response(reservation_id)

        if admin_response:
            # Admin has responded! Now resume the graph
            decision = admin_response["decision"]
            admin_name = admin_response.get("admin_name", "Admin")
            reason = admin_response.get("reason")

            print("\n" + "=" * 60)
            print("  ADMIN HAS RESPONDED!")
            print("=" * 60)

            # Resume the workflow with admin's decision
            result = workflow.resume_after_admin(
                reservation_id=reservation_id,
                decision=decision,
                admin_name=admin_name,
                reason=reason
            )

            response = result.get("response", "")
            if response:
                print(f"\nBot: {response}")
                print("")

            # Reset shared state so main loop knows we're done waiting
            shared_state["waiting_for_admin"] = False
            shared_state["current_reservation_id"] = None
            print("You: ", end="", flush=True)  # Show new prompt

            stop_event.set()
            return

        time.sleep(2)  # Check every 2 seconds


def main():
    """Main chatbot loop with HITL support."""
    print_banner()

    print("Initializing HITL workflow...")

    # Initialize database and SQL agent for answering questions
    from src.database.sql_db import ParkingDatabase
    from src.rag.sql_agent import create_sql_agent

    db = ParkingDatabase()
    sql_agent = create_sql_agent(db)

    # Create HITL workflow with SQL agent
    workflow = HITLWorkflow(db=db, sql_agent=sql_agent)
    print("Ready!\n")

    thread_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Shared state between main thread and background thread
    shared_state = {
        "waiting_for_admin": False,
        "current_reservation_id": None,
    }

    admin_check_thread = None
    stop_event = None

    try:
        while True:
            # Show prompt
            if shared_state["waiting_for_admin"]:
                prompt = f"You (waiting for admin on {shared_state['current_reservation_id']})> "
            else:
                prompt = "You: "

            try:
                user_input = input(prompt).strip()
            except EOFError:
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if user_input.lower() == 'help':
                print("""
Commands:
  book parking    - Start a reservation
  check status    - Check reservation status
  reset           - Reset conversation
  quit            - Exit

For the old CLI without HITL: uv run python -m src.cli
                """)
                continue

            if user_input.lower() == 'reset':
                workflow.reset_conversation(thread_id)
                shared_state["waiting_for_admin"] = False
                shared_state["current_reservation_id"] = None
                if stop_event:
                    stop_event.set()
                print("Conversation reset.\n")
                continue

            # If waiting for admin, remind user
            if shared_state["waiting_for_admin"]:
                print(f"\nBot: I'm still waiting for admin approval on {shared_state['current_reservation_id']}.")
                print("     Please check the dashboard at http://localhost:8001/dashboard/")
                print("     Or type 'reset' to start a new conversation.\n")
                continue

            # Process message through HITL workflow
            result = workflow.process_message(user_input, thread_id)

            # Show response
            response = result.get("response", "")
            if response:
                print(f"\nBot: {response}\n")

            # Check if we hit an interrupt (waiting for admin)
            if result.get("interrupted"):
                shared_state["waiting_for_admin"] = True
                shared_state["current_reservation_id"] = result.get("reservation_id")

                print("-" * 60)
                print("  CONVERSATION PAUSED - Waiting for admin approval")
                print(f"  Reservation ID: {shared_state['current_reservation_id']}")
                print("  Go to: http://localhost:8001/dashboard/")
                print("  Click 'Approve' or 'Reject' to continue")
                print("-" * 60)
                print("")

                # Start background thread to check for admin response
                stop_event = threading.Event()
                admin_check_thread = threading.Thread(
                    target=check_for_admin_response,
                    args=(workflow, shared_state["current_reservation_id"], thread_id, stop_event, shared_state),
                    daemon=True
                )
                admin_check_thread.start()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    finally:
        if stop_event:
            stop_event.set()
        print("\nChatbot session ended.")


if __name__ == "__main__":
    main()
