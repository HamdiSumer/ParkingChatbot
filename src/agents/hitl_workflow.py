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

import re
from src.database.sql_db import ParkingDatabase
from src.admin.admin_service import AdminService
from src.rag.llm_provider import create_llm
from src.utils.logging import logger


def parse_flexible_time(time_str: str, base_date: datetime = None) -> Optional[datetime]:
    """Parse various time formats into datetime.

    Handles:
    - "tomorrow", "today"
    - "13:45", "2pm", "14:00"
    - "2026-03-01 13:45"
    - "tomorrow 2pm"
    """
    if not time_str:
        return None

    time_str = time_str.strip().lower()
    base_date = base_date or datetime.now()

    # Try standard format first
    try:
        return datetime.strptime(time_str, "%Y-%m-%d %H:%M")
    except ValueError:
        pass

    # Handle relative dates
    target_date = base_date.date()
    if "tomorrow" in time_str:
        from datetime import timedelta
        target_date = (base_date + timedelta(days=1)).date()
        time_str = time_str.replace("tomorrow", "").strip()
    elif "today" in time_str:
        time_str = time_str.replace("today", "").strip()

    # Extract time component
    time_patterns = [
        (r'(\d{1,2}):(\d{2})', lambda m: (int(m.group(1)), int(m.group(2)))),  # 13:45
        (r'(\d{1,2})\s*(am|pm)', lambda m: (int(m.group(1)) + (12 if m.group(2) == 'pm' and int(m.group(1)) != 12 else 0), 0)),  # 2pm
        (r'(\d{1,2})h(\d{2})?', lambda m: (int(m.group(1)), int(m.group(2) or 0))),  # 14h00
    ]

    for pattern, extractor in time_patterns:
        match = re.search(pattern, time_str)
        if match:
            try:
                hour, minute = extractor(match)
                return datetime.combine(target_date, datetime.min.time().replace(hour=hour, minute=minute))
            except (ValueError, AttributeError):
                continue

    # If only date keywords, use noon as default time
    if target_date != base_date.date():
        return datetime.combine(target_date, datetime.min.time().replace(hour=12, minute=0))

    return None


def parse_time_range(time_str: str, base_date: datetime = None) -> tuple:
    """Parse a time range like '13:45 - 17:45' into (start, end) datetimes."""
    base_date = base_date or datetime.now()

    # Common range separators
    separators = [' - ', ' to ', '-', '–', 'to', 'until', 'till']

    for sep in separators:
        if sep in time_str.lower():
            parts = time_str.lower().split(sep.lower() if sep.lower() in time_str.lower() else sep)
            if len(parts) == 2:
                start = parse_flexible_time(parts[0].strip(), base_date)
                end = parse_flexible_time(parts[1].strip(), base_date)

                # If only times provided (no date), use base_date
                if start and end:
                    return start, end

    return None, None


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

    def __init__(self, db: ParkingDatabase = None, sql_agent=None):
        self.db = db or ParkingDatabase()
        self.admin_service = AdminService(self.db)
        self.llm = create_llm(temperature=0.3)
        self.sql_agent = sql_agent  # For answering data questions
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
        """Determine user intent using LLM agent."""
        msg = state.get("current_message", "")

        # If already in reservation flow, check if user is providing field data or asking a new question
        if state.get("conversation_type") == "reservation" and state.get("next_expected_field"):
            if self._is_field_response(msg, state.get("next_expected_field")):
                return state  # Continue reservation flow
            else:
                # User is asking a new question - break out of reservation
                logger.info("Agent detected new question during reservation flow, breaking out")
                state["conversation_type"] = "general"
                state["next_expected_field"] = None
                # Fall through to re-classify intent

        # Use LLM to classify intent
        intent = self._classify_intent(msg)
        logger.info(f"Agent classified intent as: {intent}")

        if intent == "status_check":
            state["conversation_type"] = "status_check"
        elif intent == "reservation":
            state["conversation_type"] = "reservation"
            if not state.get("collected_fields"):
                state["collected_fields"] = []
            if not state.get("required_fields"):
                state["required_fields"] = ["name", "surname", "car_number", "parking_id", "start_time", "end_time"]
            if not state.get("reservation_data"):
                state["reservation_data"] = {}
        else:
            # "question" or "general" - handle as general chat
            state["conversation_type"] = "general"

        return state

    def _classify_intent(self, message: str) -> str:
        """Use LLM to classify user intent.

        Returns:
            One of: "reservation", "status_check", "question", "general"
        """
        msg_lower = message.lower()

        # Fast path: Check for explicit reservation keywords first
        reservation_phrases = [
            "make a reservation", "make reservation", "book a", "book parking",
            "reserve a", "reserve parking", "i want to book", "i'd like to book",
            "i would like to book", "i want to reserve", "i'd like to reserve",
            "need to book", "need to reserve", "want to park"
        ]
        if any(phrase in msg_lower for phrase in reservation_phrases):
            logger.info("Fast path: detected reservation intent")
            return "reservation"

        # Fast path: Status check
        if "status" in msg_lower or ("check" in msg_lower and "reservation" in msg_lower):
            if "RES_" in message.upper() or "booking" in msg_lower:
                return "status_check"

        # Fast path: Greetings
        greetings = ["hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye"]
        if msg_lower.strip() in greetings or len(msg_lower.split()) <= 2 and any(g in msg_lower for g in greetings):
            return "general"

        # For ambiguous cases, use LLM
        prompt = f"""Classify this parking chatbot message. Reply with ONLY one word.

RESERVATION - User wants to CREATE a new booking (book, reserve, park)
QUESTION - Asking about info, prices, availability, follow-ups
STATUS - Checking existing reservation status
GENERAL - Greetings, thanks, chitchat

Examples:
"how many spaces left" → QUESTION
"I want to book a spot" → RESERVATION
"hi I want to make reservation" → RESERVATION
"check my reservation" → STATUS
"thanks" → GENERAL

Message: "{message}"
Classification:"""

        try:
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
            answer = answer.strip().upper().split()[0] if answer.strip() else ""

            # Map response - check RESERVATION first for this case
            if "RESERVATION" in answer:
                return "reservation"
            elif "STATUS" in answer:
                return "status_check"
            elif "GENERAL" in answer:
                return "general"
            else:
                return "question"

        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return "question"

    def _is_field_response(self, message: str, expected_field: str) -> bool:
        """Determine if message is providing field data or asking a new question.

        Uses simple heuristics first, then LLM as fallback for ambiguous cases.
        """
        msg = message.strip()
        msg_lower = msg.lower()

        # Short inputs (1-3 words) without question marks are almost always field data
        word_count = len(msg.split())
        if word_count <= 3 and not msg.endswith("?"):
            # Check it's not a clear command/question
            question_starters = ["how", "what", "where", "when", "why", "can", "is", "are", "do", "does"]
            if not any(msg_lower.startswith(q) for q in question_starters):
                return True  # It's field data

        # Clear questions - break out of reservation
        if msg.endswith("?"):
            return False

        # Question patterns - break out
        question_patterns = [
            "how many", "how much", "what is", "what are", "where is",
            "tell me", "show me", "i want to know", "can you"
        ]
        if any(pattern in msg_lower for pattern in question_patterns):
            return False

        # For longer inputs, use LLM to decide
        if word_count > 3:
            prompt = f"""Is this user input providing their {expected_field}, or asking a different question?

Input: "{message}"
Expected field: {expected_field}

Reply FIELD or QUESTION:"""
            try:
                response = self.llm.invoke(prompt)
                answer = response.content if hasattr(response, "content") else str(response)
                return "FIELD" in answer.upper()
            except Exception as e:
                logger.error(f"Field check failed: {e}")

        # Default: assume it's field data if not clearly a question
        return True

    def _intent_router(self, state: HITLState) -> str:
        """Route based on conversation type."""
        conv_type = state.get("conversation_type", "general")
        if conv_type == "reservation":
            return "collect"
        elif conv_type == "status_check":
            return "status"
        return "general"

    def _collect_reservation_node(self, state: HITLState) -> HITLState:
        """Collect reservation info from user with flexible parsing."""
        msg = state.get("current_message", "").strip()
        msg_lower = msg.lower()

        # Initialize data structures if needed
        if "reservation_data" not in state or state["reservation_data"] is None:
            state["reservation_data"] = {}
        if "collected_fields" not in state or state["collected_fields"] is None:
            state["collected_fields"] = []

        # Try to extract any data from the current message proactively
        collected = state.get("collected_fields", [])
        data = state.get("reservation_data", {})

        # Extract time range if present (e.g., "13:45 - 17:45")
        if "start_time" not in collected or "end_time" not in collected:
            start_dt, end_dt = parse_time_range(msg)
            if start_dt and end_dt:
                data["start_time"] = start_dt.isoformat()
                data["end_time"] = end_dt.isoformat()
                if "start_time" not in collected:
                    collected.append("start_time")
                if "end_time" not in collected:
                    collected.append("end_time")
                logger.info(f"Extracted time range: {start_dt} to {end_dt}")

        # Extract date mentions (e.g., "tomorrow", "for tomorrow")
        if "start_time" not in collected:
            if "tomorrow" in msg_lower or "today" in msg_lower:
                # Store for later - we still need specific times
                base_time = parse_flexible_time(msg)
                if base_time:
                    data["_temp_date"] = base_time.date().isoformat()
                    logger.info(f"Stored temp date: {data['_temp_date']}")

        # Extract parking location
        if "parking_id" not in collected:
            parking_ids = ["downtown_1", "airport_1", "riverside_1"]
            for pid in parking_ids:
                if pid in msg_lower or pid.replace("_", " ") in msg_lower:
                    data["parking_id"] = pid
                    collected.append("parking_id")
                    logger.info(f"Extracted parking_id: {pid}")
                    break
            # Also check for location names without _1
            location_map = {"downtown": "downtown_1", "airport": "airport_1", "riverside": "riverside_1"}
            for name, pid in location_map.items():
                if name in msg_lower and "parking_id" not in collected:
                    data["parking_id"] = pid
                    collected.append("parking_id")
                    logger.info(f"Extracted parking_id from name: {pid}")
                    break

        # If expecting a specific field, try to store the value
        expected_field = state.get("next_expected_field")
        if expected_field and msg:
            if expected_field == "name" and "name" not in collected:
                # Accept any non-empty input as name
                data["name"] = msg.split()[0].title()  # Take first word
                collected.append("name")
            elif expected_field == "surname" and "surname" not in collected:
                data["surname"] = msg.split()[0].title()
                collected.append("surname")
            elif expected_field == "car_number" and "car_number" not in collected:
                # Accept alphanumeric input as car number
                car_num = ''.join(msg.upper().split())
                if car_num:
                    data["car_number"] = car_num
                    collected.append("car_number")
            elif expected_field == "parking_id" and "parking_id" not in collected:
                # Already handled above, but accept direct input too
                data["parking_id"] = msg.strip()
                collected.append("parking_id")
            elif expected_field == "start_time" and "start_time" not in collected:
                # Use flexible parsing
                temp_date = data.get("_temp_date")
                base = datetime.fromisoformat(temp_date) if temp_date else datetime.now()
                dt = parse_flexible_time(msg, base)
                if dt:
                    data["start_time"] = dt.isoformat()
                    collected.append("start_time")
            elif expected_field == "end_time" and "end_time" not in collected:
                temp_date = data.get("_temp_date")
                base = datetime.fromisoformat(temp_date) if temp_date else datetime.now()
                dt = parse_flexible_time(msg, base)
                if dt:
                    data["end_time"] = dt.isoformat()
                    collected.append("end_time")

        # Update state
        state["reservation_data"] = data
        state["collected_fields"] = collected

        # Find next missing field
        required = state.get("required_fields", ["name", "surname", "car_number", "parking_id", "start_time", "end_time"])
        missing = [f for f in required if f not in collected]

        if missing:
            next_field = missing[0]
            state["next_expected_field"] = next_field

            prompts = {
                "name": "Great! Let's book your parking spot. What's your **first name**?",
                "surname": "And your **last name**?",
                "car_number": "What's your **car registration/plate number**?",
                "parking_id": "Which parking location?\n• **downtown_1** - Downtown Parking\n• **airport_1** - Airport Parking\n• **riverside_1** - Riverside Parking",
                "start_time": "When do you want to **start**? (e.g., 'tomorrow 2pm' or '2026-03-01 14:00')",
                "end_time": "And when will you **leave**? (e.g., '5pm' or '17:00')",
            }
            state["chatbot_response"] = prompts.get(next_field, f"Please provide your {next_field}")
        else:
            # All collected!
            state["next_expected_field"] = None
            # Clean up temp data
            if "_temp_date" in data:
                del data["_temp_date"]

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
        """Handle general conversation and questions using SQL agent."""
        msg = state.get("current_message", "")
        msg_lower = msg.lower()

        # Check if this is an availability question
        is_availability_question = any(kw in msg_lower for kw in [
            "how many", "available", "spaces left", "spots left", "spaces available"
        ])

        if is_availability_question:
            # Check if query is vague (missing location or time)
            has_location = any(loc in msg_lower for loc in ["downtown", "airport", "riverside"])
            has_time = any(time_word in msg_lower for time_word in [
                "today", "tomorrow", "morning", "afternoon", "evening", "now",
                ":", "am", "pm", "hour"
            ]) or any(char.isdigit() for char in msg)

            # If vague, ask for clarification
            if not has_location and not has_time:
                state["chatbot_response"] = (
                    "To check availability accurately, I need a bit more info:\n\n"
                    "**Which location?**\n"
                    "• downtown_1 - Downtown Parking\n"
                    "• airport_1 - Airport Parking\n"
                    "• riverside_1 - Riverside Parking\n\n"
                    "**When do you need parking?**\n"
                    "For example: 'tomorrow 2pm to 5pm' or 'now'\n\n"
                    "Or I can show you current overall availability if you just want a quick overview!"
                )
                return state

        # For data questions - query the database
        if state.get("conversation_type") == "general":
            is_data_question = any(kw in msg_lower for kw in [
                "how many", "available", "price", "cost", "open", "spaces", "left",
                "show", "list", "overview"
            ])

            if is_data_question and self.sql_agent:
                try:
                    result = self.sql_agent.invoke({"input": msg})
                    sql_output = result.get("output", "")

                    if sql_output:
                        # Synthesize a natural response - MUST use the exact data
                        synthesis_prompt = f"""Convert this database result into a friendly response.

User question: "{msg}"
Database result: {sql_output}

IMPORTANT: Use the EXACT numbers from the database result. Do NOT make up any numbers.
If the result shows 3 spaces, say 3. If it shows 0, say 0.
Mention that availability may change based on reservations if relevant.

Give a brief, friendly 1-2 sentence response using the actual data:"""

                        response = self.llm.invoke(synthesis_prompt)
                        answer = response.content if hasattr(response, "content") else str(response)
                        # Remove quotes if present
                        answer = answer.strip().strip('"').strip("'")
                        state["chatbot_response"] = answer
                        return state

                except Exception as e:
                    logger.error(f"SQL agent query failed: {e}")

        # Fallback to general LLM response (for non-data questions)
        response = self.llm.invoke(
            f"You are a helpful parking assistant. The user said: '{msg}'. "
            f"Give a brief, helpful response. Do NOT make up any numbers or data."
        )
        answer = response.content if hasattr(response, "content") else str(response)
        state["chatbot_response"] = answer.strip().strip('"').strip("'")
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
        config = {"configurable": {"thread_id": thread_id}}

        # Check if there's existing state in the checkpointer
        existing_state = self.graph.get_state(config)

        if existing_state.values:
            # Continue existing conversation - only pass new input
            input_data = {
                "current_message": message,
                "thread_id": thread_id,
                "chatbot_response": "",  # Reset for new turn
            }
        else:
            # New conversation - initialize full state
            input_data = HITLState(
                messages=[],
                current_message=message,
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

        try:
            # This will run until an interrupt or END
            result = self.graph.invoke(input_data, config)

            # Check if we hit an interrupt (waiting for admin)
            graph_state = self.graph.get_state(config)

            if graph_state.next:  # There are pending nodes = we're interrupted
                # Graph is paused at wait_for_admin
                logger.info(f"Graph interrupted at: {graph_state.next}")

                return {
                    "response": result.get("chatbot_response", ""),
                    "thread_id": thread_id,
                    "reservation_id": result.get("reservation_id"),
                    "waiting_for_admin": True,
                    "interrupted": True,
                    "message": "Conversation paused - waiting for admin approval",
                }

            # Normal completion
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
        config = {"configurable": {"thread_id": thread_id}}
        graph_state = self.graph.get_state(config)
        if graph_state.values:
            return graph_state.values.get("waiting_for_admin", False)
        return False

    def reset_conversation(self, thread_id: str):
        """Reset a conversation by clearing checkpointer state."""
        # Clear from checkpointer by updating with empty state
        config = {"configurable": {"thread_id": thread_id}}
        try:
            # Update state to reset all fields
            self.graph.update_state(
                config,
                {
                    "conversation_type": "general",
                    "collected_fields": [],
                    "reservation_data": {},
                    "next_expected_field": None,
                    "waiting_for_admin": False,
                    "reservation_id": None,
                }
            )
        except Exception:
            pass  # Thread might not exist yet
        logger.info(f"Conversation {thread_id} reset")
