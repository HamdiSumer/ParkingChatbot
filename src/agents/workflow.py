"""LangGraph workflow for the parking chatbot with ReAct agent."""
from typing import Literal, Optional
import re
import uuid
from datetime import datetime
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.agents.state import ConversationState, create_empty_state
from src.agents.tools import ToolRegistry
from src.agents.prompts import (
    AGENT_DECISION_PROMPT,
    DIRECT_RESPONSE_PROMPT,
    SYNTHESIS_PROMPT,
    format_conversation_history,
    format_tool_results,
)
from src.rag.retriever import ParkingRAGRetriever
from src.database.sql_db import ParkingDatabase
from src.guardrails.filter import DataProtectionFilter
from src.rag.llm_provider import create_llm
from src.utils.logging import logger


class ParkingChatbotWorkflow:
    """LangGraph-based workflow with ReAct agent for intelligent tool selection."""

    def __init__(
        self,
        rag_retriever: ParkingRAGRetriever,
        db: ParkingDatabase,
        guard_rails: DataProtectionFilter,
        sql_agent=None,
    ):
        """Initialize the chatbot workflow.

        Args:
            rag_retriever: RAG retriever for static data.
            db: SQL database for dynamic data.
            guard_rails: Data protection filter.
            sql_agent: SQL agent for dynamic queries.
        """
        self.rag_retriever = rag_retriever
        self.db = db
        self.guard_rails = guard_rails
        self.sql_agent = sql_agent
        self.llm = create_llm(temperature=0.3)

        # Initialize tool registry
        vector_store = rag_retriever.vector_store if rag_retriever else None
        self.tools = ToolRegistry(vector_store=vector_store, sql_agent=sql_agent)

        self.workflow = self._build_workflow()
        self._conversation_state: Optional[ConversationState] = None
        logger.info("Parking chatbot workflow initialized with ReAct agent")

    def _build_workflow(self):
        """Build the LangGraph workflow with ReAct agent.

        Returns:
            Compiled workflow graph.
        """
        workflow = StateGraph(ConversationState)

        # Add nodes
        workflow.add_node("safety_check", self._safety_check_node)
        workflow.add_node("agent_decide", self._agent_decide_node)
        workflow.add_node("vector_search", self._vector_search_node)
        workflow.add_node("sql_query", self._sql_query_node)
        workflow.add_node("direct_response", self._direct_response_node)
        workflow.add_node("synthesize", self._synthesize_node)
        workflow.add_node("collect_reservation", self._collect_reservation_node)
        workflow.add_node("admin_review", self._admin_review_node)
        workflow.add_node("output_filter", self._output_filter_node)

        # Start with safety check
        workflow.add_edge(START, "safety_check")

        # Safety router
        workflow.add_conditional_edges(
            "safety_check",
            self._safety_router,
            {
                "unsafe": "output_filter",  # Go directly to output with error message
                "safe": "agent_decide",
            },
        )

        # Agent decision router - decides which tool to use
        workflow.add_conditional_edges(
            "agent_decide",
            self._agent_action_router,
            {
                "vector_search": "vector_search",
                "sql_query": "sql_query",
                "direct_response": "direct_response",
                "start_reservation": "collect_reservation",
                "synthesize": "synthesize",
            },
        )

        # After vector search, check if more tools needed
        workflow.add_conditional_edges(
            "vector_search",
            self._tool_continuation_router,
            {
                "need_more": "agent_decide",
                "sufficient": "synthesize",
            },
        )

        # After SQL query, check if more tools needed
        workflow.add_conditional_edges(
            "sql_query",
            self._tool_continuation_router,
            {
                "need_more": "agent_decide",
                "sufficient": "synthesize",
            },
        )

        # Direct response goes straight to output
        workflow.add_edge("direct_response", "output_filter")

        # Synthesize goes to output
        workflow.add_edge("synthesize", "output_filter")

        # Reservation collection
        workflow.add_conditional_edges(
            "collect_reservation",
            self._reservation_router,
            {
                "collecting": "output_filter",
                "ready": "admin_review",
            },
        )

        # Admin review to output
        workflow.add_edge("admin_review", "output_filter")

        # Output filter ends
        workflow.add_edge("output_filter", END)

        return workflow.compile()

    # ==================== NODES ====================

    def _safety_check_node(self, state: ConversationState) -> ConversationState:
        """Check message for safety issues."""
        is_safe, issue = self.guard_rails.check_safety(state.current_message)

        if not is_safe:
            state.safety_issue_detected = True
            state.safety_issue_details = issue
            state.chatbot_response = f"I cannot process this request. {issue}"
            logger.warning(f"Safety issue detected: {issue}")
        else:
            state.safety_issue_detected = False

        return state

    def _agent_decide_node(self, state: ConversationState) -> ConversationState:
        """ReAct agent decision node - decides what action to take."""
        logger.info(f"Agent deciding action for: {state.current_message[:50]}...")

        # Check iteration limit
        if state.iteration_count >= state.max_iterations:
            logger.warning("Max iterations reached, forcing synthesize")
            state.agent_decision = "synthesize"
            return state

        state.iteration_count += 1

        # Build the prompt
        prompt = AGENT_DECISION_PROMPT.format(
            tools_description=self.tools.get_tools_description(),
            conversation_history=format_conversation_history(state.messages),
            tool_results=format_tool_results(state.tool_results),
            user_message=state.current_message,
        )

        try:
            # Get agent decision from LLM
            response = self.llm.invoke(prompt)
            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)

            # Parse the response
            action = self._parse_agent_response(response_text, state)
            state.agent_decision = action
            logger.info(f"Agent decision: {action}")

        except Exception as e:
            logger.error(f"Agent decision failed: {e}")
            # Default to direct response on error
            state.agent_decision = "direct_response"

        return state

    def _parse_agent_response(self, response: str, state: ConversationState) -> str:
        """Parse agent response to extract action.

        Args:
            response: Raw LLM response.
            state: Current state (for context).

        Returns:
            Action string.
        """
        msg_lower = state.current_message.lower().strip()

        # FIRST: Check for obvious greetings/simple messages (override LLM for reliability)
        greetings = ["hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye", "ok", "okay", "sure", "yes", "no"]
        msg_words = msg_lower.split()

        # If message is just a greeting or very short, use direct response
        if len(msg_words) <= 2:
            if any(g == msg_lower or g in msg_words for g in greetings):
                logger.info("Detected greeting/simple message, using direct_response")
                return "direct_response"
            # Very short messages that aren't parking-related
            parking_words = ["park", "parking", "space", "spot", "book", "reserve", "price", "cost", "available", "location", "where"]
            if not any(pw in msg_lower for pw in parking_words):
                return "direct_response"

        # Check for reservation intent
        reservation_words = ["book", "reserve", "reservation", "make a booking"]
        if any(rw in msg_lower for rw in reservation_words):
            return "start_reservation"

        # If we have tool results already, synthesize
        if state.tools_used:
            return "synthesize"

        # Check for real-time data queries BEFORE LLM parsing (more reliable)
        realtime_words = ["available", "how many", "price", "cost", "open", "status", "current", "now"]
        if any(rw in msg_lower for rw in realtime_words):
            logger.info("Detected real-time query, using sql_query")
            return "sql_query"

        response_lower = response.lower()

        # Look for ACTION: line in LLM response
        action_match = re.search(r'action:\s*(\w+)', response_lower)
        if action_match:
            action = action_match.group(1)
            valid_actions = ["vector_search", "sql_query", "direct_response", "start_reservation", "synthesize"]
            if action in valid_actions:
                return action

        # Fallback: analyze response content
        if "direct_response" in response_lower or "direct response" in response_lower:
            return "direct_response"
        if "sql_query" in response_lower or "sql query" in response_lower:
            return "sql_query"
        if "vector_search" in response_lower or "vector search" in response_lower:
            return "vector_search"
        if "reservation" in response_lower:
            return "start_reservation"

        # Default to vector search for parking/info questions
        return "vector_search"

    def _vector_search_node(self, state: ConversationState) -> ConversationState:
        """Execute vector search on parking knowledge base."""
        logger.info("Executing vector search")

        result = self.rag_retriever.vector_search_only(
            state.current_message, k=3, use_reranking=True
        )

        if result.get("success"):
            state.vector_search_results = result.get("documents", [])
            state.tool_results["vector_search"] = result.get("formatted", "")
            state.tools_used.append("vector_search")
            state.should_include_sources = True
            logger.info(f"Vector search returned {len(state.vector_search_results)} documents")
        else:
            logger.warning(f"Vector search failed: {result.get('error')}")

        return state

    def _sql_query_node(self, state: ConversationState) -> ConversationState:
        """Execute SQL query for real-time data."""
        logger.info("Executing SQL query")

        if not self.sql_agent:
            logger.warning("SQL agent not available")
            return state

        try:
            result = self.sql_agent.invoke({"input": state.current_message})
            output = result.get("output", "")

            if output:
                state.sql_query_results = output
                state.tool_results["sql_query"] = output
                state.tools_used.append("sql_query")
                state.should_include_sources = True
                logger.info("SQL query executed successfully")
            else:
                logger.info("SQL query returned no results")

        except Exception as e:
            logger.error(f"SQL query failed: {e}")

        return state

    def _direct_response_node(self, state: ConversationState) -> ConversationState:
        """Generate direct response without tool use."""
        logger.info("Generating direct response (no tools)")

        prompt = DIRECT_RESPONSE_PROMPT.format(
            conversation_history=format_conversation_history(state.messages),
            user_message=state.current_message,
        )

        try:
            response = self.llm.invoke(prompt)
            if hasattr(response, "content"):
                state.chatbot_response = response.content
            else:
                state.chatbot_response = str(response)

            # Clean thinking tags if present
            state.chatbot_response = self._clean_response(state.chatbot_response)

        except Exception as e:
            logger.error(f"Direct response failed: {e}")
            state.chatbot_response = "Hello! I'm your parking assistant. How can I help you today?"

        # No sources for direct responses
        state.should_include_sources = False
        state.response_sources = []

        return state

    def _synthesize_node(self, state: ConversationState) -> ConversationState:
        """Synthesize final response from gathered tool results."""
        logger.info("Synthesizing response from tool results")

        # Build context from tool results
        context_parts = []

        if state.tool_results.get("vector_search"):
            context_parts.append(
                "=== PARKING INFORMATION ===\n" + state.tool_results["vector_search"]
            )

        if state.tool_results.get("sql_query"):
            context_parts.append(
                "=== REAL-TIME DATA ===\n" + state.tool_results["sql_query"]
            )

        context = "\n\n".join(context_parts) if context_parts else "No information gathered."

        prompt = SYNTHESIS_PROMPT.format(
            context=context,
            user_message=state.current_message,
            conversation_history=format_conversation_history(state.messages),
        )

        try:
            response = self.llm.invoke(prompt)
            if hasattr(response, "content"):
                state.chatbot_response = response.content
            else:
                state.chatbot_response = str(response)

            state.chatbot_response = self._clean_response(state.chatbot_response)

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            state.chatbot_response = "I found some information but couldn't generate a proper response."

        # Set sources if tools were used
        if state.should_include_sources and state.vector_search_results:
            state.response_sources = state.vector_search_results

        return state

    def _output_filter_node(self, state: ConversationState) -> ConversationState:
        """Apply output filtering and finalize response."""
        if state.chatbot_response:
            filtered_response, was_modified = self.guard_rails.filter_output(
                state.chatbot_response
            )
            if was_modified:
                logger.info("Response was filtered by output guardrails")
            state.chatbot_response = filtered_response

        # Add to message history
        state.messages.append({"role": "user", "content": state.current_message})
        state.messages.append({
            "role": "assistant",
            "content": state.chatbot_response,
            "tools_used": state.tools_used,
        })

        logger.info("Conversation step completed")
        return state

    def _collect_reservation_node(self, state: ConversationState) -> ConversationState:
        """Collect user information for reservation."""
        logger.info("Collecting reservation information")
        state.conversation_type = "reservation"

        # If we're waiting for a specific field, extract it
        if state.next_expected_field and state.current_message:
            self._extract_and_store_field(
                state, state.next_expected_field, state.current_message
            )

        # Check what fields are still missing
        fields_to_ask = [
            f for f in state.required_fields if f not in state.collected_fields
        ]

        if not fields_to_ask:
            state.pending_admin_review = True
            state.next_expected_field = None
            collected_summary = self._format_collected_data(state)
            state.chatbot_response = (
                f"Great! I have all your information:\n{collected_summary}\n\n"
                "An administrator will review your reservation shortly."
            )
        else:
            next_field = fields_to_ask[0]
            state.next_expected_field = next_field

            field_prompts = {
                "name": "What is your first name?",
                "surname": "What is your last name?",
                "car_number": "What is your car registration number? (e.g., ABC-123)",
                "parking_id": f"Which parking space would you like to book? Available: {self._get_available_parking_ids()}",
                "start_time": "When would you like to start parking? (YYYY-MM-DD HH:MM)",
                "end_time": "When would you like to end parking? (YYYY-MM-DD HH:MM)",
            }
            state.chatbot_response = field_prompts.get(
                next_field, "What information can I help you with?"
            )

        return state

    def _admin_review_node(self, state: ConversationState) -> ConversationState:
        """Submit reservation to admin for review."""
        logger.info("Submitting reservation for admin review")

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
                state.chatbot_response = (
                    f"Your reservation {res_id} is pending admin approval. "
                    "You will be notified once it's reviewed."
                )
                logger.info(f"Reservation {res_id} created")
            else:
                state.chatbot_response = "Failed to create reservation. Please try again."

        return state

    # ==================== ROUTERS ====================

    def _safety_router(self, state: ConversationState) -> Literal["unsafe", "safe"]:
        """Route based on safety check."""
        return "unsafe" if state.safety_issue_detected else "safe"

    def _agent_action_router(
        self, state: ConversationState
    ) -> Literal["vector_search", "sql_query", "direct_response", "start_reservation", "synthesize"]:
        """Route based on agent decision."""
        decision = state.agent_decision
        valid = ["vector_search", "sql_query", "direct_response", "start_reservation", "synthesize"]
        return decision if decision in valid else "direct_response"

    def _tool_continuation_router(
        self, state: ConversationState
    ) -> Literal["need_more", "sufficient"]:
        """Determine if agent needs more tools."""
        # Check iteration limit
        if state.iteration_count >= state.max_iterations:
            return "sufficient"

        # Simple heuristic: if we have both vector and sql results, we're done
        if "vector_search" in state.tools_used and "sql_query" in state.tools_used:
            return "sufficient"

        # Check if the query might need both tools
        msg_lower = state.current_message.lower()
        needs_realtime = any(
            kw in msg_lower
            for kw in ["available", "price", "how many", "status", "open", "current"]
        )
        needs_static = any(
            kw in msg_lower
            for kw in ["where", "location", "address", "rule", "policy", "how to"]
        )

        # If query seems to need both and we only have one, get more
        if needs_realtime and needs_static:
            if "vector_search" in state.tools_used and "sql_query" not in state.tools_used:
                state.needs_more_info = True
                return "need_more"
            if "sql_query" in state.tools_used and "vector_search" not in state.tools_used:
                state.needs_more_info = True
                return "need_more"

        return "sufficient"

    def _reservation_router(
        self, state: ConversationState
    ) -> Literal["collecting", "ready"]:
        """Route based on reservation data collection status."""
        missing = [f for f in state.required_fields if f not in state.collected_fields]
        return "ready" if not missing else "collecting"

    # ==================== HELPERS ====================

    def _clean_response(self, response: str) -> str:
        """Remove thinking tags from reasoning model outputs."""
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        cleaned = "\n".join(
            line.strip() for line in cleaned.split("\n") if line.strip()
        )
        return cleaned.strip()

    def _extract_and_store_field(
        self, state: ConversationState, field: str, message: str
    ) -> bool:
        """Extract a field value from user message and store it."""
        value = message.strip()

        if field == "name" or field == "surname":
            cleaned = re.sub(r"[^a-zA-Z\s\-]", "", value).strip()
            if len(cleaned) >= 2:
                state.reservation_data[field] = cleaned.title()
                state.collected_fields.append(field)
                return True

        elif field == "car_number":
            cleaned = re.sub(r"[^a-zA-Z0-9\-\s]", "", value).strip().upper()
            if len(cleaned) >= 3:
                state.reservation_data[field] = cleaned
                state.collected_fields.append(field)
                return True

        elif field == "parking_id":
            available_ids = self._get_available_parking_ids_list()
            value_lower = value.lower()
            for pid in available_ids:
                if pid.lower() in value_lower or value_lower in pid.lower():
                    state.reservation_data[field] = pid
                    state.collected_fields.append(field)
                    return True
            if value in available_ids:
                state.reservation_data[field] = value
                state.collected_fields.append(field)
                return True

        elif field in ["start_time", "end_time"]:
            parsed = self._parse_datetime(value)
            if parsed:
                state.reservation_data[field] = parsed
                state.collected_fields.append(field)
                return True

        return False

    def _parse_datetime(self, value: str) -> Optional[datetime]:
        """Parse datetime from user input."""
        formats = [
            "%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M",
            "%d-%m-%Y %H:%M",
            "%d/%m/%Y %H:%M",
            "%Y-%m-%d",
            "%d-%m-%Y",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(value.strip(), fmt)
            except ValueError:
                continue
        return None

    def _get_available_parking_ids(self) -> str:
        """Get available parking IDs as a formatted string."""
        try:
            session = self.db.get_session()
            from src.database.sql_db import ParkingSpace

            spaces = session.query(ParkingSpace).all()
            session.close()
            return (
                ", ".join([f"{s.id} ({s.name})" for s in spaces])
                or "No parking spaces available"
            )
        except Exception:
            return "downtown_1, airport_1, riverside_1"

    def _get_available_parking_ids_list(self) -> list:
        """Get available parking IDs as a list."""
        try:
            session = self.db.get_session()
            from src.database.sql_db import ParkingSpace

            spaces = session.query(ParkingSpace).all()
            session.close()
            return [s.id for s in spaces]
        except Exception:
            return ["downtown_1", "airport_1", "riverside_1"]

    def _format_collected_data(self, state: ConversationState) -> str:
        """Format collected reservation data for display."""
        data = state.reservation_data
        lines = [
            f"- Name: {data.get('name', 'N/A')} {data.get('surname', '')}",
            f"- Car: {data.get('car_number', 'N/A')}",
            f"- Parking: {data.get('parking_id', 'N/A')}",
            f"- Start: {data.get('start_time', 'N/A')}",
            f"- End: {data.get('end_time', 'N/A')}",
        ]
        return "\n".join(lines)

    # ==================== PUBLIC API ====================

    def invoke(self, user_message: str) -> dict:
        """Process a user message through the workflow.

        Args:
            user_message: User's message.

        Returns:
            Dictionary with response and metadata.
        """
        # Use persistent state or create new one
        if self._conversation_state is None:
            self._conversation_state = create_empty_state()

        # Reset agent state for new query (but keep conversation history)
        self._conversation_state.reset_agent_state()
        self._conversation_state.current_message = user_message

        # Run workflow
        final_state = self.workflow.invoke(self._conversation_state)

        # Update persistent state from workflow result
        if isinstance(final_state, dict):
            self._conversation_state.chatbot_response = final_state.get(
                "chatbot_response", ""
            )
            self._conversation_state.response_sources = final_state.get(
                "response_sources", []
            )
            self._conversation_state.conversation_type = final_state.get(
                "conversation_type", "general"
            )
            self._conversation_state.safety_issue_detected = final_state.get(
                "safety_issue_detected", False
            )
            self._conversation_state.messages = final_state.get("messages", [])
            self._conversation_state.collected_fields = final_state.get(
                "collected_fields", []
            )
            self._conversation_state.reservation_data = final_state.get(
                "reservation_data", {}
            )
            self._conversation_state.next_expected_field = final_state.get(
                "next_expected_field"
            )
            self._conversation_state.should_include_sources = final_state.get(
                "should_include_sources", False
            )
            self._conversation_state.tools_used = final_state.get("tools_used", [])
        else:
            self._conversation_state = final_state

        # Check if reservation was submitted
        if self._conversation_state.pending_admin_review:
            response = {
                "response": self._conversation_state.chatbot_response,
                "sources": (
                    self._conversation_state.response_sources
                    if self._conversation_state.should_include_sources
                    else []
                ),
                "type": self._conversation_state.conversation_type,
                "safety_issue": self._conversation_state.safety_issue_detected,
                "messages": self._conversation_state.messages,
                "tools_used": self._conversation_state.tools_used,
                "reservation_complete": True,
            }
            self._conversation_state = None
            return response

        return {
            "response": self._conversation_state.chatbot_response,
            "sources": (
                self._conversation_state.response_sources
                if self._conversation_state.should_include_sources
                else []
            ),
            "type": self._conversation_state.conversation_type,
            "safety_issue": self._conversation_state.safety_issue_detected,
            "messages": self._conversation_state.messages,
            "tools_used": self._conversation_state.tools_used,
            "collecting_reservation": self._conversation_state.conversation_type
            == "reservation",
        }

    def reset_conversation(self):
        """Reset conversation state to start fresh."""
        self._conversation_state = None
        logger.info("Conversation state reset")
