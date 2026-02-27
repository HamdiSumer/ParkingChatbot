"""Human-in-the-Loop (HITL) LangGraph Workflow.

This implements TRUE HITL where the conversation PAUSES and WAITS
for admin approval before continuing.

Key features:
- Graph INTERRUPTS when waiting for admin (doesn't just continue)
- Thread-based persistence allows resuming conversations
- Admin dashboard triggers graph resume
- Bot "wakes up" when admin responds
"""
import uuid
import json
from datetime import datetime
from typing import TypedDict, Literal, Optional, List, Dict, Any, Annotated
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage, AIMessage

from src.database.sql_db import ParkingDatabase
from src.admin.admin_service import AdminService
from src.rag.llm_provider import create_llm
from src.utils.logging import logger


# ==================== STATE DEFINITION ====================

class HITLState(TypedDict, total=False):
    """State for HITL workflow."""
    # Conversation
    messages: List[Dict[str, str]]
    current_message: str
    thread_id: str

    # Flow control
    conversation_type: Literal["general", "reservation", "status_check"]
    waiting_for_admin: bool  # TRUE = graph is interrupted

    # Reservation data
    reservation_data: Dict[str, Any]
    collected_fields: List[str]
    required_fields: List[str]
    next_expected_field: Optional[str]
    reservation_id: Optional[str]

    # Admin response (filled when admin approves/rejects)
    admin_decision: Optional[Literal["approve", "reject"]]
    admin_name: Optional[str]
    admin_reason: Optional[str]

    # Response
    chatbot_response: str


# ==================== PERSISTENCE FOR INTERRUPTED THREADS ====================

class ThreadStore:
    """Simple file-based store for interrupted threads.

    In production, use Redis or a proper database.
    """

    def __init__(self, store_path: str = "./data/threads"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

    def save_waiting_thread(self, thread_id: str, reservation_id: str, state: dict):
        """Save thread that's waiting for admin."""
        data = {
            "thread_id": thread_id,
            "reservation_id": reservation_id,
            "state": state,
            "created_at": datetime.utcnow().isoformat(),
            "status": "waiting_for_admin",
            "admin_decision": None,
            "admin_name": None,
            "admin_reason": None,
        }
        file_path = self.store_path / f"{reservation_id}.json"
        with open(file_path, "w") as f:
            json.dump(data, f, default=str)
        logger.info(f"Thread {thread_id} saved, waiting for admin on {reservation_id}")

    def get_waiting_thread(self, reservation_id: str) -> Optional[dict]:
        """Get thread waiting for admin response."""
        file_path = self.store_path / f"{reservation_id}.json"
        if file_path.exists():
            with open(file_path, "r") as f:
                return json.load(f)
        return None

    def update_with_admin_decision(
        self,
        reservation_id: str,
        decision: str,
        admin_name: str,
        reason: str = None
    ):
        """Update thread with admin's decision (called by dashboard)."""
        file_path = self.store_path / f"{reservation_id}.json"
        if file_path.exists():
            with open(file_path, "r") as f:
                data = json.load(f)

            data["status"] = "admin_responded"
            data["admin_decision"] = decision
            data["admin_name"] = admin_name
            data["admin_reason"] = reason
            data["responded_at"] = datetime.utcnow().isoformat()

            with open(file_path, "w") as f:
                json.dump(data, f, default=str)

            logger.info(f"Thread for {reservation_id} updated with admin decision: {decision}")
            return True
        return False

    def check_admin_response(self, reservation_id: str) -> Optional[dict]:
        """Check if admin has responded to a thread."""
        thread = self.get_waiting_thread(reservation_id)
        if thread and thread.get("status") == "admin_responded":
            return {
                "decision": thread.get("admin_decision"),
                "admin_name": thread.get("admin_name"),
                "reason": thread.get("admin_reason"),
            }
        return None

    def remove_waiting_thread(self, reservation_id: str):
        """Remove thread after processing."""
        file_path = self.store_path / f"{reservation_id}.json"
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Thread for {reservation_id} removed")

    def get_all_waiting(self) -> List[dict]:
        """Get all threads waiting for admin."""
        threads = []
        for file_path in self.store_path.glob("*.json"):
            with open(file_path, "r") as f:
                data = json.load(f)
                if data.get("status") == "waiting_for_admin":
                    threads.append(data)
        return threads


# ==================== HITL WORKFLOW ====================

class HITLWorkflow:
    """LangGraph workflow with TRUE Human-in-the-Loop.

    When a reservation is submitted:
    1. Bot says "Please wait while I get approval..."
    2. Graph INTERRUPTS (pauses, does NOT continue)
    3. Admin sees request on dashboard
    4. Admin clicks Approve/Reject
    5. Dashboard calls resume_thread()
    6. Graph RESUMES from where it paused
    7. Bot says "Great news! Approved!" or "Sorry, rejected."
    """

    def __init__(self, db: ParkingDatabase = None):
        self.db = db or ParkingDatabase()
        self.admin_service = AdminService(self.db)
        self.llm = create_llm(temperature=0.3)
        self.thread_store = ThreadStore()

        # Checkpointer for thread persistence
        self.checkpointer = MemorySaver()

        # Build the graph
        self.graph = self._build_graph()

        # Active conversations (thread_id -> state)
        self._conversations: Dict[str, HITLState] = {}

        logger.info("HITL Workflow initialized with interrupt support")

    def _build_graph(self) -> StateGraph:
        """Build the HITL workflow graph.

        Graph structure:
        START → route_intent → [collect_info / general_chat / check_status]
                                     ↓
                              (if all collected)
                                     ↓
                              submit_to_admin
                                     ↓
                              wait_for_admin  ← INTERRUPT HAPPENS HERE
                                     ↓
                              process_admin_response
                                     ↓
                                    END
        """
        workflow = StateGraph(HITLState)

        # Nodes
        workflow.add_node("route_intent", self._route_intent_node)
        workflow.add_node("collect_reservation", self._collect_reservation_node)
        workflow.add_node("general_chat", self._general_chat_node)
        workflow.add_node("check_status", self._check_status_node)
        workflow.add_node("submit_to_admin", self._submit_to_admin_node)
        workflow.add_node("wait_for_admin", self._wait_for_admin_node)
        workflow.add_node("process_admin_response", self._process_admin_response_node)
        workflow.add_node("finalize", self._finalize_node)

        # Edges
        workflow.add_edge(START, "route_intent")

        # Route based on intent
        workflow.add_conditional_edges(
            "route_intent",
            self._intent_router,
            {
                "collect": "collect_reservation",
                "general": "general_chat",
                "status": "check_status",
            }
        )

        # Collection routes
        workflow.add_conditional_edges(
            "collect_reservation",
            self._collection_router,
            {
                "continue": "finalize",
                "submit": "submit_to_admin",
            }
        )

        # General and status go to finalize
        workflow.add_edge("general_chat", "finalize")
        workflow.add_edge("check_status", "finalize")

        # Submit to admin then wait
        workflow.add_edge("submit_to_admin", "wait_for_admin")

        # Wait for admin - this is where INTERRUPT happens
        # After interrupt resumes, go to process response
        workflow.add_edge("wait_for_admin", "process_admin_response")

        # Process response then finalize
        workflow.add_edge("process_admin_response", "finalize")

        # Finalize ends
        workflow.add_edge("finalize", END)

        return workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["wait_for_admin"],  # Interrupt BEFORE this node
        )

    # ==================== NODES ====================

    def _route_intent_node(self, state: HITLState) -> HITLState:
        """Determine user intent."""
        msg = state.get("current_message", "").lower()

        # If already in reservation flow, continue
        if state.get("conversation_type") == "reservation" and state.get("next_expected_field"):
            return state

        # Check for status check
        if any(kw in msg for kw in ["status", "check", "my reservation", "res_"]):
            state["conversation_type"] = "status_check"
            return state

        # Check for reservation intent
        if any(kw in msg for kw in ["book", "reserve", "reservation", "parking"]):
            state["conversation_type"] = "reservation"
            if not state.get("collected_fields"):
                state["collected_fields"] = []
            if not state.get("required_fields"):
                state["required_fields"] = ["name", "surname", "car_number", "parking_id", "start_time", "end_time"]
            if not state.get("reservation_data"):
                state["reservation_data"] = {}
            return state

        # Default to general
        state["conversation_type"] = "general"
        return state

    def _intent_router(self, state: HITLState) -> str:
        """Route based on conversation type."""
        conv_type = state.get("conversation_type", "general")
        if conv_type == "reservation":
            return "collect"
        elif conv_type == "status_check":
            return "status"
        return "general"

    def _collect_reservation_node(self, state: HITLState) -> HITLState:
        """Collect reservation info from user."""
        # If expecting a field, extract it
        if state.get("next_expected_field") and state.get("current_message"):
            field = state["next_expected_field"]
            value = state["current_message"].strip()

            # Store the value
            if field in ["name", "surname"]:
                state["reservation_data"][field] = value.title()
                state["collected_fields"].append(field)
            elif field == "car_number":
                state["reservation_data"][field] = value.upper()
                state["collected_fields"].append(field)
            elif field == "parking_id":
                state["reservation_data"][field] = value
                state["collected_fields"].append(field)
            elif field in ["start_time", "end_time"]:
                try:
                    dt = datetime.strptime(value, "%Y-%m-%d %H:%M")
                    state["reservation_data"][field] = dt.isoformat()
                    state["collected_fields"].append(field)
                except ValueError:
                    state["chatbot_response"] = "Invalid format. Please use YYYY-MM-DD HH:MM"
                    return state

        # Find next missing field
        required = state.get("required_fields", [])
        collected = state.get("collected_fields", [])
        missing = [f for f in required if f not in collected]

        if missing:
            next_field = missing[0]
            state["next_expected_field"] = next_field

            prompts = {
                "name": "What is your first name?",
                "surname": "What is your last name?",
                "car_number": "What is your car registration number?",
                "parking_id": "Which parking spot? (downtown_1, airport_1, riverside_1)",
                "start_time": "Start time? (YYYY-MM-DD HH:MM)",
                "end_time": "End time? (YYYY-MM-DD HH:MM)",
            }
            state["chatbot_response"] = prompts.get(next_field, f"Please provide {next_field}")
        else:
            # All collected!
            state["next_expected_field"] = None

        return state

    def _collection_router(self, state: HITLState) -> str:
        """Route based on collection completeness."""
        if state.get("next_expected_field") is None and state.get("collected_fields"):
            required = state.get("required_fields", [])
            collected = state.get("collected_fields", [])
            if all(f in collected for f in required):
                return "submit"
        return "continue"

    def _general_chat_node(self, state: HITLState) -> HITLState:
        """Handle general conversation."""
        response = self.llm.invoke(
            f"You are a parking assistant. Respond briefly: {state['current_message']}"
        )
        state["chatbot_response"] = response.content if hasattr(response, "content") else str(response)
        return state

    def _check_status_node(self, state: HITLState) -> HITLState:
        """Check reservation status."""
        import re
        msg = state.get("current_message", "")
        match = re.search(r'RES_[A-Za-z0-9]+', msg, re.IGNORECASE)

        if match:
            res_id = match.group(0).upper()
            status = self.admin_service.get_reservation_status(res_id)
            if status:
                s = status["status"].upper()
                if s == "CONFIRMED":
                    s = "APPROVED"
                state["chatbot_response"] = f"Reservation {res_id}: **{s}**"
            else:
                state["chatbot_response"] = f"Reservation {res_id} not found."
        else:
            state["chatbot_response"] = "Please provide your reservation ID (e.g., RES_ABC123)"

        return state

    def _submit_to_admin_node(self, state: HITLState) -> HITLState:
        """Submit reservation to database and notify admin."""
        data = state["reservation_data"]

        # Create reservation in DB
        res_id = f"RES_{uuid.uuid4().hex[:8].upper()}"

        # Parse datetime strings back to datetime objects
        start_time = datetime.fromisoformat(data["start_time"]) if isinstance(data["start_time"], str) else data["start_time"]
        end_time = datetime.fromisoformat(data["end_time"]) if isinstance(data["end_time"], str) else data["end_time"]

        success = self.db.create_reservation(
            res_id=res_id,
            user_name=data["name"],
            user_surname=data["surname"],
            car_number=data["car_number"],
            parking_id=data["parking_id"],
            start_time=start_time,
            end_time=end_time,
        )

        if success:
            state["reservation_id"] = res_id
            state["waiting_for_admin"] = True

            # Save thread state for later resume
            self.thread_store.save_waiting_thread(
                thread_id=state.get("thread_id", "default"),
                reservation_id=res_id,
                state=dict(state),
            )

            # This message will be shown BEFORE the interrupt
            state["chatbot_response"] = (
                f"I've submitted your reservation **{res_id}**.\n\n"
                f"**Please wait while I get approval from the administrator...**\n\n"
                f"(The administrator has been notified. I'll let you know as soon as they respond.)"
            )

            logger.info(f"Reservation {res_id} submitted, waiting for admin approval")
        else:
            state["chatbot_response"] = "Failed to create reservation. Please try again."
            state["waiting_for_admin"] = False

        return state

    def _wait_for_admin_node(self, state: HITLState) -> HITLState:
        """This node is where the INTERRUPT happens.

        The graph is configured to interrupt BEFORE this node.
        When we reach here, it means admin has responded and we're resuming.
        """
        # If we get here, admin has responded (graph was resumed)
        # The admin_decision should be populated by the resume call
        logger.info(f"Resuming after admin response: {state.get('admin_decision')}")
        return state

    def _process_admin_response_node(self, state: HITLState) -> HITLState:
        """Process the admin's decision."""
        res_id = state.get("reservation_id")
        decision = state.get("admin_decision")
        admin_name = state.get("admin_name", "Admin")

        if decision == "approve":
            # Update database
            self.admin_service.approve_reservation(res_id, admin_name, None)
            state["chatbot_response"] = (
                f"**Great news!** Your reservation **{res_id}** has been **APPROVED**!\n\n"
                f"Approved by: {admin_name}\n"
                f"You're all set. Have a great parking experience!"
            )
            logger.info(f"Reservation {res_id} APPROVED by {admin_name}")

        elif decision == "reject":
            reason = state.get("admin_reason", "No reason provided")
            self.admin_service.reject_reservation(res_id, admin_name, reason)
            state["chatbot_response"] = (
                f"**Unfortunately**, your reservation **{res_id}** was **REJECTED**.\n\n"
                f"Reason: {reason}\n\n"
                f"Please try booking a different time or parking location."
            )
            logger.info(f"Reservation {res_id} REJECTED by {admin_name}: {reason}")

        else:
            state["chatbot_response"] = f"Reservation {res_id} is still pending."

        # Clean up
        state["waiting_for_admin"] = False
        self.thread_store.remove_waiting_thread(res_id)

        return state

    def _finalize_node(self, state: HITLState) -> HITLState:
        """Finalize and store message history."""
        if state.get("current_message"):
            if "messages" not in state:
                state["messages"] = []
            state["messages"].append({"role": "user", "content": state["current_message"]})
        if state.get("chatbot_response"):
            if "messages" not in state:
                state["messages"] = []
            state["messages"].append({"role": "assistant", "content": state["chatbot_response"]})
        return state

    # ==================== PUBLIC API ====================

    def process_message(self, message: str, thread_id: str = None) -> Dict[str, Any]:
        """Process a user message.

        Args:
            message: User's message
            thread_id: Thread ID for conversation continuity

        Returns:
            Dict with response and metadata
        """
        thread_id = thread_id or str(uuid.uuid4())

        # Get or create state
        if thread_id in self._conversations:
            state = self._conversations[thread_id]
        else:
            state = HITLState(
                messages=[],
                current_message="",
                thread_id=thread_id,
                conversation_type="general",
                waiting_for_admin=False,
                reservation_data={},
                collected_fields=[],
                required_fields=["name", "surname", "car_number", "parking_id", "start_time", "end_time"],
                next_expected_field=None,
                reservation_id=None,
                admin_decision=None,
                admin_name=None,
                admin_reason=None,
                chatbot_response="",
            )

        state["current_message"] = message
        state["thread_id"] = thread_id

        # Run the graph
        config = {"configurable": {"thread_id": thread_id}}

        try:
            # This will run until an interrupt or END
            result = self.graph.invoke(state, config)

            # Check if we hit an interrupt (waiting for admin)
            graph_state = self.graph.get_state(config)

            if graph_state.next:  # There are pending nodes = we're interrupted
                # Graph is paused at wait_for_admin
                logger.info(f"Graph interrupted at: {graph_state.next}")

                # Update conversation state
                self._conversations[thread_id] = result

                return {
                    "response": result.get("chatbot_response", ""),
                    "thread_id": thread_id,
                    "reservation_id": result.get("reservation_id"),
                    "waiting_for_admin": True,
                    "interrupted": True,
                    "message": "Conversation paused - waiting for admin approval",
                }

            # Normal completion
            self._conversations[thread_id] = result

            return {
                "response": result.get("chatbot_response", ""),
                "thread_id": thread_id,
                "reservation_id": result.get("reservation_id"),
                "waiting_for_admin": False,
                "interrupted": False,
            }

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "response": f"Error: {str(e)}",
                "thread_id": thread_id,
                "waiting_for_admin": False,
                "interrupted": False,
            }

    def resume_after_admin(
        self,
        reservation_id: str,
        decision: Literal["approve", "reject"],
        admin_name: str,
        reason: str = None,
    ) -> Dict[str, Any]:
        """Resume an interrupted conversation after admin responds.

        This is called by the dashboard when admin clicks Approve/Reject.

        Args:
            reservation_id: The reservation ID
            decision: "approve" or "reject"
            admin_name: Name of admin
            reason: Rejection reason (if rejecting)

        Returns:
            The bot's response to show to user
        """
        # Find the waiting thread
        thread_data = self.thread_store.get_waiting_thread(reservation_id)

        if not thread_data:
            logger.warning(f"No waiting thread for {reservation_id}")
            return {"error": f"No pending conversation for {reservation_id}"}

        thread_id = thread_data["thread_id"]
        config = {"configurable": {"thread_id": thread_id}}

        # Update state with admin's decision
        self.graph.update_state(
            config,
            {
                "admin_decision": decision,
                "admin_name": admin_name,
                "admin_reason": reason,
                "waiting_for_admin": False,
            },
        )

        # Resume the graph (continue from interrupt)
        result = self.graph.invoke(None, config)

        # Update conversation state
        self._conversations[thread_id] = result

        logger.info(f"Resumed conversation for {reservation_id} after {decision}")

        return {
            "response": result.get("chatbot_response", ""),
            "thread_id": thread_id,
            "reservation_id": reservation_id,
            "decision": decision,
            "resumed": True,
        }

    def get_waiting_conversations(self) -> List[Dict]:
        """Get all conversations waiting for admin."""
        return self.thread_store.get_all_waiting()

    def is_waiting_for_admin(self, thread_id: str) -> bool:
        """Check if a thread is waiting for admin."""
        state = self._conversations.get(thread_id, {})
        return state.get("waiting_for_admin", False)

    def reset_conversation(self, thread_id: str):
        """Reset a conversation."""
        if thread_id in self._conversations:
            del self._conversations[thread_id]
        logger.info(f"Conversation {thread_id} reset")
