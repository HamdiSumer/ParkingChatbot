"""Multi-Agent LangGraph Workflow with Human-in-the-Loop.

This module implements a proper agent-to-agent communication system where:
- Agent 1 (User Agent): Collects reservation info from user
- Agent 2 (Admin Agent): Actively notifies admin and waits for approval
- HITL Interrupt: Graph pauses until admin responds
- Proactive Notification: User is notified immediately when status changes
"""
import uuid
import asyncio
from datetime import datetime
from typing import TypedDict, Literal, Optional, List, Dict, Any, Annotated
from dataclasses import dataclass, field

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.database.sql_db import ParkingDatabase
from src.admin.admin_service import AdminService
from src.rag.llm_provider import create_llm
from src.utils.logging import logger


# ==================== SHARED STATE ====================

class ReservationData(TypedDict, total=False):
    """Reservation data collected from user."""
    name: str
    surname: str
    car_number: str
    parking_id: str
    start_time: datetime
    end_time: datetime


class MultiAgentState(TypedDict, total=False):
    """Shared state between agents.

    This state flows through the entire multi-agent workflow,
    allowing Agent 1 to hand off to Agent 2 seamlessly.
    """
    # Conversation
    messages: List[Dict[str, str]]
    current_message: str

    # Agent tracking
    current_agent: Literal["user_agent", "admin_agent", "none"]
    handoff_reason: str

    # User Agent state
    conversation_type: str
    reservation_data: ReservationData
    collected_fields: List[str]
    required_fields: List[str]
    next_expected_field: Optional[str]

    # Reservation state
    reservation_id: Optional[str]
    reservation_status: Literal["collecting", "pending_admin", "approved", "rejected"]

    # Admin Agent state
    admin_notified: bool
    admin_notification_sent_at: Optional[datetime]
    awaiting_admin_response: bool
    admin_decision: Optional[Literal["approve", "reject"]]
    admin_name: Optional[str]
    admin_notes: Optional[str]
    rejection_reason: Optional[str]

    # Response
    chatbot_response: str
    pending_user_notification: Optional[str]  # Proactive notification for user


def create_initial_state() -> MultiAgentState:
    """Create initial multi-agent state."""
    return MultiAgentState(
        messages=[],
        current_message="",
        current_agent="user_agent",
        handoff_reason="",
        conversation_type="general",
        reservation_data={},
        collected_fields=[],
        required_fields=["name", "surname", "car_number", "parking_id", "start_time", "end_time"],
        next_expected_field=None,
        reservation_id=None,
        reservation_status="collecting",
        admin_notified=False,
        admin_notification_sent_at=None,
        awaiting_admin_response=False,
        admin_decision=None,
        admin_name=None,
        admin_notes=None,
        rejection_reason=None,
        chatbot_response="",
        pending_user_notification=None,
    )


# ==================== NOTIFICATION SERVICE ====================

class ActiveNotificationService:
    """Service for actively pushing notifications to admin.

    This service implements the "Active Push" model - it doesn't just
    update a database, it actively notifies the admin via multiple channels.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        email_config: Optional[Dict] = None,
        slack_webhook: Optional[str] = None,
    ):
        self.webhook_url = webhook_url
        self.email_config = email_config
        self.slack_webhook = slack_webhook
        self._pending_notifications: Dict[str, Dict] = {}
        self._admin_responses: Dict[str, Dict] = {}
        logger.info("ActiveNotificationService initialized")

    async def notify_admin(
        self,
        reservation_id: str,
        reservation_data: ReservationData,
        callback_url: str = None,
    ) -> Dict[str, Any]:
        """Actively notify admin about new reservation.

        This sends notifications via all configured channels:
        - Webhook (REST API callback)
        - Email (SMTP)
        - Slack (webhook)

        Args:
            reservation_id: The reservation ID
            reservation_data: Reservation details
            callback_url: URL for admin to respond

        Returns:
            Dict with notification status and channels used
        """
        notification = {
            "id": str(uuid.uuid4()),
            "reservation_id": reservation_id,
            "type": "new_reservation",
            "data": dict(reservation_data),
            "timestamp": datetime.utcnow().isoformat(),
            "callback_url": callback_url or f"/api/admin/respond/{reservation_id}",
            "status": "pending",
        }

        self._pending_notifications[reservation_id] = notification

        channels_notified = []

        # 1. Console notification (always)
        self._print_admin_notification(notification)
        channels_notified.append("console")

        # 2. Webhook notification
        if self.webhook_url:
            try:
                await self._send_webhook(notification)
                channels_notified.append("webhook")
            except Exception as e:
                logger.error(f"Webhook notification failed: {e}")

        # 3. Email notification
        if self.email_config:
            try:
                await self._send_email(notification)
                channels_notified.append("email")
            except Exception as e:
                logger.error(f"Email notification failed: {e}")

        # 4. Slack notification
        if self.slack_webhook:
            try:
                await self._send_slack(notification)
                channels_notified.append("slack")
            except Exception as e:
                logger.error(f"Slack notification failed: {e}")

        logger.info(f"Admin notified via: {channels_notified}")

        return {
            "success": True,
            "notification_id": notification["id"],
            "channels": channels_notified,
            "reservation_id": reservation_id,
        }

    def register_admin_response(
        self,
        reservation_id: str,
        decision: Literal["approve", "reject"],
        admin_name: str,
        notes: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        """Register admin's response to a notification.

        This is called when admin responds (via API, email reply, Slack button, etc.)
        """
        self._admin_responses[reservation_id] = {
            "decision": decision,
            "admin_name": admin_name,
            "notes": notes,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Remove from pending
        if reservation_id in self._pending_notifications:
            del self._pending_notifications[reservation_id]

        logger.info(f"Admin response registered: {reservation_id} -> {decision}")

    def get_admin_response(self, reservation_id: str) -> Optional[Dict]:
        """Check if admin has responded to a reservation."""
        return self._admin_responses.get(reservation_id)

    def has_pending_notification(self, reservation_id: str) -> bool:
        """Check if notification is still pending."""
        return reservation_id in self._pending_notifications

    async def _send_webhook(self, notification: Dict):
        """Send notification via webhook."""
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.webhook_url,
                json=notification,
                timeout=10.0,
            )
            response.raise_for_status()

    async def _send_email(self, notification: Dict):
        """Send notification via email."""
        # Placeholder - implement with actual SMTP
        logger.info(f"[EMAIL] Would send email for reservation {notification['reservation_id']}")
        # In production, use smtplib or a service like SendGrid

    async def _send_slack(self, notification: Dict):
        """Send notification via Slack webhook."""
        import httpx

        slack_message = {
            "text": f"New Parking Reservation Request!",
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "New Reservation Request"}
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*ID:* {notification['reservation_id']}"},
                        {"type": "mrkdwn", "text": f"*Name:* {notification['data'].get('name', 'N/A')} {notification['data'].get('surname', '')}"},
                        {"type": "mrkdwn", "text": f"*Car:* {notification['data'].get('car_number', 'N/A')}"},
                        {"type": "mrkdwn", "text": f"*Parking:* {notification['data'].get('parking_id', 'N/A')}"},
                    ]
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Approve"},
                            "style": "primary",
                            "action_id": f"approve_{notification['reservation_id']}",
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Reject"},
                            "style": "danger",
                            "action_id": f"reject_{notification['reservation_id']}",
                        },
                    ]
                }
            ]
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.slack_webhook,
                json=slack_message,
                timeout=10.0,
            )
            response.raise_for_status()

    def _print_admin_notification(self, notification: Dict):
        """Print notification to console."""
        print("\n" + "=" * 70)
        print("  ADMIN NOTIFICATION - NEW RESERVATION REQUEST")
        print("=" * 70)
        print(f"  Reservation ID: {notification['reservation_id']}")
        print(f"  Name: {notification['data'].get('name', 'N/A')} {notification['data'].get('surname', '')}")
        print(f"  Car: {notification['data'].get('car_number', 'N/A')}")
        print(f"  Parking: {notification['data'].get('parking_id', 'N/A')}")
        print(f"  Time: {notification['timestamp']}")
        print("-" * 70)
        print(f"  Respond at: {notification['callback_url']}")
        print("  Or use admin_cli.py: approve {id} / reject {id}")
        print("=" * 70 + "\n")


# ==================== MULTI-AGENT WORKFLOW ====================

class MultiAgentWorkflow:
    """LangGraph-based multi-agent workflow with proper handoff.

    This implements the "Active Push" model where:
    1. User Agent collects reservation info
    2. User Agent hands off to Admin Agent
    3. Admin Agent actively notifies admin
    4. Graph interrupts until admin responds
    5. Admin Agent processes response
    6. User is proactively notified
    """

    def __init__(
        self,
        db: ParkingDatabase = None,
        notification_service: ActiveNotificationService = None,
    ):
        self.db = db or ParkingDatabase()
        self.admin_service = AdminService(self.db)
        self.notification_service = notification_service or ActiveNotificationService()
        self.llm = create_llm(temperature=0.3)

        # Build the multi-agent graph
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

        # Track active conversations
        self._conversation_states: Dict[str, MultiAgentState] = {}

        logger.info("MultiAgentWorkflow initialized with HITL support")

    def _build_graph(self) -> StateGraph:
        """Build the multi-agent LangGraph workflow.

        Graph structure:

        START → check_pending_notification → [notify_user / user_agent]
                                                     ↓
                                            route_user_agent
                                           ↙        ↓        ↘
                              collect_info    general_chat    check_status
                                   ↓
                              handoff_to_admin
                                   ↓
                              admin_notify
                                   ↓
                              admin_wait (INTERRUPT)
                                   ↓
                              admin_process
                                   ↓
                              notify_user → END
        """
        workflow = StateGraph(MultiAgentState)

        # Nodes
        workflow.add_node("check_pending_notification", self._check_pending_notification_node)
        workflow.add_node("notify_user_proactively", self._notify_user_proactively_node)
        workflow.add_node("user_agent", self._user_agent_node)
        workflow.add_node("collect_reservation_info", self._collect_reservation_info_node)
        workflow.add_node("general_chat", self._general_chat_node)
        workflow.add_node("check_status", self._check_status_node)
        workflow.add_node("handoff_to_admin", self._handoff_to_admin_node)
        workflow.add_node("admin_notify", self._admin_notify_node)
        workflow.add_node("admin_wait", self._admin_wait_node)
        workflow.add_node("admin_process", self._admin_process_node)
        workflow.add_node("finalize_response", self._finalize_response_node)

        # Entry point - check if there's a pending notification for user
        workflow.add_edge(START, "check_pending_notification")

        # Route based on pending notification
        workflow.add_conditional_edges(
            "check_pending_notification",
            self._route_pending_notification,
            {
                "notify_user": "notify_user_proactively",
                "continue": "user_agent",
            }
        )

        # After proactive notification, continue to user agent
        workflow.add_edge("notify_user_proactively", "user_agent")

        # User agent routes to appropriate handler
        workflow.add_conditional_edges(
            "user_agent",
            self._route_user_agent,
            {
                "collect_info": "collect_reservation_info",
                "general_chat": "general_chat",
                "check_status": "check_status",
            }
        )

        # Collection routes based on completeness
        workflow.add_conditional_edges(
            "collect_reservation_info",
            self._route_collection,
            {
                "continue_collecting": "finalize_response",
                "handoff": "handoff_to_admin",
            }
        )

        # General chat and status check go to finalize
        workflow.add_edge("general_chat", "finalize_response")
        workflow.add_edge("check_status", "finalize_response")

        # Admin handoff flow
        workflow.add_edge("handoff_to_admin", "admin_notify")
        workflow.add_edge("admin_notify", "admin_wait")

        # Admin wait routes based on response availability
        workflow.add_conditional_edges(
            "admin_wait",
            self._route_admin_wait,
            {
                "wait": "finalize_response",  # Return to user while waiting
                "process": "admin_process",  # Admin has responded
            }
        )

        # Admin process goes to finalize
        workflow.add_edge("admin_process", "finalize_response")

        # Finalize ends
        workflow.add_edge("finalize_response", END)

        return workflow.compile(checkpointer=self.checkpointer)

    # ==================== NODES ====================

    def _check_pending_notification_node(self, state: MultiAgentState) -> MultiAgentState:
        """Check if there's a pending notification for the user."""
        # Check if any of user's reservations have been approved/rejected
        if state.get("reservation_id"):
            response = self.notification_service.get_admin_response(state["reservation_id"])
            if response and not state.get("pending_user_notification"):
                decision = response["decision"]
                if decision == "approve":
                    state["pending_user_notification"] = (
                        f"Great news! Your reservation **{state['reservation_id']}** "
                        f"has been **APPROVED** by {response['admin_name']}!"
                    )
                else:
                    state["pending_user_notification"] = (
                        f"Unfortunately, your reservation **{state['reservation_id']}** "
                        f"was **REJECTED**. Reason: {response.get('reason', 'No reason provided')}"
                    )
                state["admin_decision"] = decision
                state["reservation_status"] = "approved" if decision == "approve" else "rejected"

        return state

    def _route_pending_notification(
        self, state: MultiAgentState
    ) -> Literal["notify_user", "continue"]:
        """Route based on whether there's a pending notification."""
        if state.get("pending_user_notification"):
            return "notify_user"
        return "continue"

    def _notify_user_proactively_node(self, state: MultiAgentState) -> MultiAgentState:
        """Proactively notify user about reservation status change."""
        notification = state.get("pending_user_notification", "")

        # Prepend the notification to the response
        state["chatbot_response"] = notification + "\n\n"
        state["pending_user_notification"] = None  # Clear after sending

        logger.info(f"Proactively notified user: {notification[:50]}...")
        return state

    def _user_agent_node(self, state: MultiAgentState) -> MultiAgentState:
        """User Agent - determines intent and routes accordingly."""
        state["current_agent"] = "user_agent"
        msg_lower = state.get("current_message", "").lower()

        # Detect intent
        if any(kw in msg_lower for kw in ["book", "reserve", "reservation", "parking"]):
            if state.get("conversation_type") != "reservation":
                state["conversation_type"] = "reservation"
                state["collected_fields"] = []

        elif any(kw in msg_lower for kw in ["status", "check", "my reservation"]):
            state["conversation_type"] = "status_check"

        elif state.get("conversation_type") == "reservation" and state.get("next_expected_field"):
            # Continue collecting
            pass
        else:
            state["conversation_type"] = "general"

        return state

    def _route_user_agent(
        self, state: MultiAgentState
    ) -> Literal["collect_info", "general_chat", "check_status"]:
        """Route based on user intent."""
        conv_type = state.get("conversation_type", "general")

        if conv_type == "reservation":
            return "collect_info"
        elif conv_type == "status_check":
            return "check_status"
        return "general_chat"

    def _collect_reservation_info_node(self, state: MultiAgentState) -> MultiAgentState:
        """Collect reservation information from user."""
        # If waiting for a field, extract it
        if state.get("next_expected_field") and state.get("current_message"):
            field = state["next_expected_field"]
            value = state["current_message"].strip()

            # Simple extraction - in production, use NLP
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
                    state["reservation_data"][field] = dt
                    state["collected_fields"].append(field)
                except ValueError:
                    state["chatbot_response"] = f"Invalid format. Please use YYYY-MM-DD HH:MM"
                    return state

        # Check what's still needed
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
                "parking_id": "Which parking spot would you like? (e.g., downtown_1, airport_1)",
                "start_time": "When do you want to start? (Format: YYYY-MM-DD HH:MM)",
                "end_time": "When do you want to end? (Format: YYYY-MM-DD HH:MM)",
            }
            state["chatbot_response"] = prompts.get(next_field, f"Please provide {next_field}")
        else:
            # All fields collected - ready for handoff
            state["next_expected_field"] = None
            state["reservation_status"] = "pending_admin"
            state["handoff_reason"] = "All reservation details collected"

        return state

    def _route_collection(
        self, state: MultiAgentState
    ) -> Literal["continue_collecting", "handoff"]:
        """Route based on collection status."""
        if state.get("reservation_status") == "pending_admin":
            return "handoff"
        return "continue_collecting"

    def _general_chat_node(self, state: MultiAgentState) -> MultiAgentState:
        """Handle general conversation."""
        # Use LLM for general chat
        response = self.llm.invoke(
            f"You are a helpful parking assistant. Respond briefly to: {state['current_message']}"
        )
        state["chatbot_response"] = response.content if hasattr(response, "content") else str(response)
        return state

    def _check_status_node(self, state: MultiAgentState) -> MultiAgentState:
        """Check reservation status."""
        # Try to find reservation ID in message or state
        res_id = state.get("reservation_id")

        if not res_id:
            import re
            match = re.search(r'RES_[A-Za-z0-9]+', state.get("current_message", ""), re.IGNORECASE)
            if match:
                res_id = match.group(0).upper()

        if res_id:
            status = self.admin_service.get_reservation_status(res_id)
            if status:
                state["chatbot_response"] = self._format_status(status)
            else:
                state["chatbot_response"] = f"Reservation {res_id} not found."
        else:
            state["chatbot_response"] = "Please provide your reservation ID (e.g., RES_ABC12345)"

        return state

    def _format_status(self, status: Dict) -> str:
        """Format reservation status for user."""
        s = status.get("status", "unknown").upper()
        if s == "CONFIRMED":
            s = "APPROVED"

        result = f"**Reservation {status['id']}**: {s}\n"
        result += f"- Parking: {status.get('parking_name', status.get('parking_id', 'N/A'))}\n"

        if s == "APPROVED":
            result += f"- Approved by: {status.get('approved_by', 'Admin')}\n"
        elif s == "REJECTED":
            result += f"- Reason: {status.get('rejection_reason', 'N/A')}\n"
        elif s == "PENDING":
            result += "- Awaiting admin approval...\n"

        return result

    def _handoff_to_admin_node(self, state: MultiAgentState) -> MultiAgentState:
        """Hand off from User Agent to Admin Agent.

        This is the critical handoff point where Agent 1 explicitly
        transfers control to Agent 2.
        """
        logger.info(f"HANDOFF: User Agent → Admin Agent ({state['handoff_reason']})")

        # Create reservation in database
        res_id = f"RES_{uuid.uuid4().hex[:8].upper()}"
        data = state["reservation_data"]

        success = self.db.create_reservation(
            res_id=res_id,
            user_name=data.get("name", ""),
            user_surname=data.get("surname", ""),
            car_number=data.get("car_number", ""),
            parking_id=data.get("parking_id", ""),
            start_time=data.get("start_time", datetime.utcnow()),
            end_time=data.get("end_time", datetime.utcnow()),
        )

        if success:
            state["reservation_id"] = res_id
            state["current_agent"] = "admin_agent"
            logger.info(f"Reservation {res_id} created, handing to Admin Agent")
        else:
            state["chatbot_response"] = "Failed to create reservation. Please try again."
            state["current_agent"] = "user_agent"

        return state

    def _admin_notify_node(self, state: MultiAgentState) -> MultiAgentState:
        """Admin Agent actively notifies administrator.

        This implements the "Active Push" - not just updating DB,
        but actively reaching out to admin.
        """
        logger.info("Admin Agent: Sending active notification to admin")

        # Run async notification in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                self.notification_service.notify_admin(
                    reservation_id=state["reservation_id"],
                    reservation_data=state["reservation_data"],
                )
            )

            state["admin_notified"] = True
            state["admin_notification_sent_at"] = datetime.utcnow()
            state["awaiting_admin_response"] = True

            logger.info(f"Admin notified via: {result['channels']}")

        finally:
            loop.close()

        return state

    def _admin_wait_node(self, state: MultiAgentState) -> MultiAgentState:
        """Wait for admin response.

        This node checks if admin has responded. In a full implementation,
        this would use LangGraph's interrupt() to pause the graph.
        """
        # Check if admin has already responded
        response = self.notification_service.get_admin_response(state["reservation_id"])

        if response:
            # Admin has responded - continue processing
            state["admin_decision"] = response["decision"]
            state["admin_name"] = response["admin_name"]
            state["admin_notes"] = response.get("notes")
            state["rejection_reason"] = response.get("reason")
            state["awaiting_admin_response"] = False
            logger.info(f"Admin response received: {response['decision']}")
        else:
            # Admin hasn't responded yet
            # In full implementation, this would use: interrupt({"reason": "awaiting_admin"})
            state["chatbot_response"] = (
                f"Your reservation **{state['reservation_id']}** has been submitted!\n\n"
                f"The administrator has been notified and will review your request shortly.\n"
                f"You'll be notified as soon as a decision is made.\n\n"
                f"You can also check the status anytime by saying 'check status {state['reservation_id']}'"
            )
            logger.info(f"Awaiting admin response for {state['reservation_id']}")

        return state

    def _route_admin_wait(
        self, state: MultiAgentState
    ) -> Literal["wait", "process"]:
        """Route based on admin response availability."""
        if state.get("admin_decision"):
            return "process"
        return "wait"

    def _admin_process_node(self, state: MultiAgentState) -> MultiAgentState:
        """Process admin's decision and update database."""
        res_id = state["reservation_id"]
        decision = state["admin_decision"]
        admin_name = state.get("admin_name", "Admin")

        if decision == "approve":
            self.admin_service.approve_reservation(
                res_id, admin_name, state.get("admin_notes")
            )
            state["reservation_status"] = "approved"
            state["chatbot_response"] = (
                f"Great news! Your reservation **{res_id}** has been **APPROVED**!\n"
                f"Approved by: {admin_name}\n"
                f"Have a great parking experience!"
            )
        else:
            reason = state.get("rejection_reason", "No reason provided")
            self.admin_service.reject_reservation(res_id, admin_name, reason)
            state["reservation_status"] = "rejected"
            state["chatbot_response"] = (
                f"Unfortunately, your reservation **{res_id}** was **REJECTED**.\n"
                f"Reason: {reason}\n"
                f"Please try booking a different time or location."
            )

        state["awaiting_admin_response"] = False
        logger.info(f"Reservation {res_id} {decision}d by {admin_name}")

        return state

    def _finalize_response_node(self, state: MultiAgentState) -> MultiAgentState:
        """Finalize the response and clean up state."""
        # Add message to history
        if state.get("current_message"):
            state["messages"].append({"role": "user", "content": state["current_message"]})
        if state.get("chatbot_response"):
            state["messages"].append({"role": "assistant", "content": state["chatbot_response"]})

        return state

    # ==================== PUBLIC API ====================

    def process_message(
        self,
        message: str,
        conversation_id: str = "default",
    ) -> Dict[str, Any]:
        """Process a user message through the multi-agent workflow.

        Args:
            message: User's message
            conversation_id: ID for conversation continuity

        Returns:
            Dict with response and metadata
        """
        # Get or create state
        if conversation_id in self._conversation_states:
            state = self._conversation_states[conversation_id]
        else:
            state = create_initial_state()

        state["current_message"] = message

        # Run the graph
        config = {"configurable": {"thread_id": conversation_id}}
        result = self.graph.invoke(state, config)

        # Update stored state
        self._conversation_states[conversation_id] = result

        return {
            "response": result.get("chatbot_response", ""),
            "reservation_id": result.get("reservation_id"),
            "status": result.get("reservation_status"),
            "current_agent": result.get("current_agent"),
            "awaiting_admin": result.get("awaiting_admin_response", False),
        }

    def register_admin_decision(
        self,
        reservation_id: str,
        decision: Literal["approve", "reject"],
        admin_name: str,
        notes: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        """Register admin's decision (called from admin CLI or API)."""
        self.notification_service.register_admin_response(
            reservation_id=reservation_id,
            decision=decision,
            admin_name=admin_name,
            notes=notes,
            reason=reason,
        )

        # The next time the user sends a message, they'll be proactively notified
        logger.info(f"Admin decision registered: {reservation_id} -> {decision}")

    def reset_conversation(self, conversation_id: str = "default"):
        """Reset a conversation state."""
        if conversation_id in self._conversation_states:
            del self._conversation_states[conversation_id]
        logger.info(f"Conversation {conversation_id} reset")
