"""LangGraph workflow for the parking chatbot."""
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from src.agents.state import ConversationState, create_empty_state
from src.rag.retriever import ParkingRAGRetriever
from src.database.sql_db import ParkingDatabase
from src.guardrails.filter import DataProtectionFilter
from src.utils.logging import logger
import uuid
from datetime import datetime


class ParkingChatbotWorkflow:
    """LangGraph-based workflow for the parking chatbot."""

    def __init__(self, rag_retriever: ParkingRAGRetriever, db: ParkingDatabase, guard_rails: DataProtectionFilter):
        """Initialize the chatbot workflow.

        Args:
            rag_retriever: RAG retriever for static data.
            db: SQL database for dynamic data.
            guard_rails: Data protection filter.
        """
        self.rag_retriever = rag_retriever
        self.db = db
        self.guard_rails = guard_rails
        self.workflow = self._build_workflow()
        logger.info("Parking chatbot workflow initialized")

    def _build_workflow(self):
        """Build the LangGraph workflow.

        Returns:
            Compiled workflow graph.
        """
        workflow = StateGraph(ConversationState)

        # Add nodes
        workflow.add_node("process_query", self._process_query_node)
        workflow.add_node("collect_reservation", self._collect_reservation_node)
        workflow.add_node("admin_review", self._admin_review_node)
        workflow.add_node("complete", self._complete_node)
        workflow.add_node("safety_check", self._safety_check_node)

        # Add edges
        workflow.add_edge(START, "safety_check")
        workflow.add_conditional_edges(
            "safety_check",
            self._safety_router,
            {
                "unsafe": END,
                "safe": "process_query",
            },
        )
        workflow.add_conditional_edges(
            "process_query",
            self._query_router,
            {
                "info": "complete",
                "reservation": "collect_reservation",
            },
        )
        workflow.add_edge("collect_reservation", "admin_review")
        workflow.add_conditional_edges(
            "admin_review",
            self._admin_router,
            {
                "approved": "complete",
                "rejected": "complete",
            },
        )
        workflow.add_edge("complete", END)

        return workflow.compile()

    def _safety_check_node(self, state: ConversationState) -> ConversationState:
        """Check message for safety issues before processing.

        Args:
            state: Current conversation state.

        Returns:
            Updated state with safety check results.
        """
        is_safe, issue = self.guard_rails.check_safety(state.current_message)

        if not is_safe:
            state.safety_issue_detected = True
            state.safety_issue_details = issue
            state.chatbot_response = f"I cannot process this request. {issue}"
            logger.warning(f"Safety issue detected: {issue}")
        else:
            state.safety_issue_detected = False

        return state

    def _safety_router(self, state: ConversationState) -> Literal["unsafe", "safe"]:
        """Route based on safety check.

        Args:
            state: Current conversation state.

        Returns:
            Routing decision.
        """
        return "unsafe" if state.safety_issue_detected else "safe"

    def _process_query_node(self, state: ConversationState) -> ConversationState:
        """Process user query using RAG.

        Args:
            state: Current conversation state.

        Returns:
            Updated state with query response.
        """
        logger.info(f"Processing query: {state.current_message[:50]}...")

        # Detect intent
        intent = self._detect_intent(state.current_message)
        state.conversation_type = intent

        if intent == "info":
            # Use RAG to answer information query
            result = self.rag_retriever.query(state.current_message)
            state.chatbot_response = result["answer"]
            state.response_sources = result["sources"]
            logger.info("Information query processed via RAG")
        elif intent == "reservation":
            state.chatbot_response = "I'll help you make a reservation. Let me collect some information."
            logger.info("Reservation intent detected")

        return state

    def _query_router(self, state: ConversationState) -> Literal["info", "reservation"]:
        """Route based on detected intent.

        Args:
            state: Current conversation state.

        Returns:
            Routing decision: "info" or "reservation".
        """
        return state.conversation_type if state.conversation_type in ["info", "reservation"] else "info"

    def _collect_reservation_node(self, state: ConversationState) -> ConversationState:
        """Collect user information for reservation.

        Args:
            state: Current conversation state.

        Returns:
            Updated state with collected data.
        """
        logger.info("Collecting reservation information")

        # Extract fields from message using simple pattern matching
        fields_to_ask = [f for f in state.required_fields if f not in state.collected_fields]

        if not fields_to_ask:
            # All required fields collected
            state.pending_admin_review = True
            state.chatbot_response = "Great! I have all your information. An administrator will review your reservation shortly."
        else:
            # Ask for the next field
            next_field = fields_to_ask[0]
            field_prompts = {
                "name": "What is your first name?",
                "surname": "What is your last name?",
                "car_number": "What is your car registration number?",
                "parking_id": "Which parking space would you like to book?",
                "start_time": "When would you like to start parking? (YYYY-MM-DD HH:MM)",
                "end_time": "When would you like to end parking? (YYYY-MM-DD HH:MM)",
            }
            state.chatbot_response = field_prompts.get(next_field, "What information can I help you with?")

        return state

    def _admin_review_node(self, state: ConversationState) -> ConversationState:
        """Submit reservation to admin for review.

        Args:
            state: Current conversation state.

        Returns:
            Updated state with admin review status.
        """
        logger.info("Submitting reservation for admin review")

        # Create reservation record with pending status
        if state.reservation_data.get("name") and state.reservation_data.get("surname"):
            res_id = f"RES_{uuid.uuid4().hex[:8].upper()}"
            success = self.db.create_reservation(
                res_id=res_id,
                user_name=state.reservation_data["name"],
                user_surname=state.reservation_data["surname"],
                car_number=state.reservation_data["car_number"],
                parking_id=state.reservation_data["parking_id"],
                start_time=state.reservation_data.get("start_time", datetime.utcnow()),
                end_time=state.reservation_data.get("end_time", datetime.utcnow()),
            )

            if success:
                state.chatbot_response = f"Your reservation {res_id} is pending admin approval. You will be notified once it's reviewed."
                logger.info(f"Reservation {res_id} created and sent for admin review")
            else:
                state.chatbot_response = "Failed to create reservation. Please try again."

        return state

    def _admin_router(self, state: ConversationState) -> Literal["approved", "rejected"]:
        """Route based on admin decision.

        Args:
            state: Current conversation state.

        Returns:
            Routing decision.
        """
        return "approved" if state.admin_decision == "approved" else "rejected"

    def _complete_node(self, state: ConversationState) -> ConversationState:
        """Complete the conversation.

        Args:
            state: Current conversation state.

        Returns:
            Final state.
        """
        # Add to message history
        state.messages.append({"role": "user", "content": state.current_message})
        state.messages.append({"role": "assistant", "content": state.chatbot_response})
        logger.info("Conversation step completed")
        return state

    def _detect_intent(self, message: str) -> str:
        """Detect user intent from message.

        Args:
            message: User message.

        Returns:
            Intent type ('info' or 'reservation').
        """
        reservation_keywords = ["book", "reserve", "reservation", "want to park", "make reservation"]
        message_lower = message.lower()

        if any(keyword in message_lower for keyword in reservation_keywords):
            return "reservation"
        return "info"

    def invoke(self, user_message: str) -> dict:
        """Process a user message through the workflow.

        Args:
            user_message: User's message.

        Returns:
            Dictionary with response and metadata.
        """
        # Create initial state
        state = create_empty_state()
        state.current_message = user_message

        # Run workflow
        final_state = self.workflow.invoke(state)

        # Handle both dict and ConversationState returns
        if isinstance(final_state, dict):
            return {
                "response": final_state.get("chatbot_response", "No response"),
                "sources": final_state.get("response_sources", []),
                "type": final_state.get("conversation_type", "info"),
                "safety_issue": final_state.get("safety_issue_detected", False),
                "messages": final_state.get("messages", []),
            }
        else:
            return {
                "response": final_state.chatbot_response,
                "sources": final_state.response_sources,
                "type": final_state.conversation_type,
                "safety_issue": final_state.safety_issue_detected,
                "messages": final_state.messages,
            }
