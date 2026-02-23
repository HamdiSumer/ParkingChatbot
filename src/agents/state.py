"""State management for the parking chatbot workflow."""
from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class ConversationState:
    """State of the conversation in the chatbot."""

    # User message history
    messages: List[dict] = field(default_factory=list)
    current_message: str = ""

    # Conversation context
    conversation_type: str = "info"  # 'info', 'reservation', 'admin'

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

    # Responses
    chatbot_response: str = ""
    response_sources: List[dict] = field(default_factory=list)

    # Admin review
    pending_admin_review: bool = False
    admin_decision: Optional[str] = None

    # Safety/Guard rails
    safety_issue_detected: bool = False
    safety_issue_details: str = ""

    # Routing
    next_step: str = "process_query"  # 'process_query', 'collect_reservation', 'admin_review', 'complete'


def create_empty_state() -> ConversationState:
    """Create an empty conversation state.

    Returns:
        Empty ConversationState instance.
    """
    return ConversationState()
