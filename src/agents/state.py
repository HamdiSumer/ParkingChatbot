"""State management for the parking chatbot workflow."""
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class ConversationState:
    """State of the conversation in the chatbot."""

    # User message history
    messages: List[dict] = field(default_factory=list)
    current_message: str = ""

    # Conversation context
    conversation_type: str = "general"  # 'general', 'reservation', 'admin'

    # === Agent Decision Tracking (NEW) ===
    agent_decision: str = ""  # Current decision: vector_search, sql_query, direct_response, start_reservation, synthesize
    tools_used: List[str] = field(default_factory=list)  # Tools called in this turn
    tool_results: Dict[str, Any] = field(default_factory=dict)  # Results from each tool
    vector_search_results: List[Any] = field(default_factory=list)  # Documents from vector search
    sql_query_results: str = ""  # Results from SQL query

    # Agent control
    should_include_sources: bool = False  # Whether to show sources in response
    iteration_count: int = 0  # Current iteration in ReAct loop
    max_iterations: int = 3  # Maximum tool calls per query
    needs_more_info: bool = False  # Does agent need another tool call?

    # Reservation data collection
    reservation_data: dict = field(
        default_factory=lambda: {
            "name": None,
            "surname": None,
            "car_number": None,
            "parking_id": None,
            "start_time": None,
            "end_time": None,
        }
    )

    # Tracking collected fields
    collected_fields: List[str] = field(default_factory=list)
    required_fields: List[str] = field(
        default_factory=lambda: ["name", "surname", "car_number", "parking_id", "start_time", "end_time"]
    )
    next_expected_field: Optional[str] = None  # The field we're waiting for from the user

    # Responses
    chatbot_response: str = ""
    response_sources: List[dict] = field(default_factory=list)

    # Admin review
    pending_admin_review: bool = False
    admin_decision: Optional[str] = None

    # Status check
    checking_status: bool = False
    status_reservation_id: Optional[str] = None

    # Safety/Guard rails
    safety_issue_detected: bool = False
    safety_issue_details: str = ""

    # Routing
    next_step: str = "agent_decide"  # 'agent_decide', 'collect_reservation', 'admin_review', 'complete'

    def reset_agent_state(self):
        """Reset agent state for a new query."""
        self.agent_decision = ""
        self.tools_used = []
        self.tool_results = {}
        self.vector_search_results = []
        self.sql_query_results = ""
        self.should_include_sources = False
        self.iteration_count = 0
        self.needs_more_info = False


def create_empty_state() -> ConversationState:
    """Create an empty conversation state.

    Returns:
        Empty ConversationState instance.
    """
    return ConversationState()
