"""LangChain Admin Agent for reservation management.

This agent handles the communication between the user chatbot and administrators.
It can submit reservations, check status, and communicate admin decisions.
"""
import httpx
from typing import Optional, Dict, Any, List
from datetime import datetime

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from src.database.sql_db import ParkingDatabase
from src.admin.admin_service import AdminService
from src.rag.llm_provider import create_llm
from src.utils.logging import logger


# ==================== ADMIN AGENT CLASS ====================

class AdminAgent:
    """LangGraph Agent for handling admin-related reservation tasks.

    This agent provides intelligent handling of reservation workflows including:
    - Submitting new reservations to the system
    - Checking reservation status
    - Processing admin approvals/rejections
    - Communicating with the REST API

    The agent uses LangGraph's ReAct pattern for decision-making.
    """

    def __init__(
        self,
        db: ParkingDatabase = None,
        api_base_url: str = "http://localhost:8001",
        use_api: bool = True,
    ):
        """Initialize the Admin Agent.

        Args:
            db: Database instance (for direct DB access when API not available).
            api_base_url: Base URL for the Admin REST API.
            use_api: Whether to use REST API (True) or direct DB access (False).
        """
        self.db = db or ParkingDatabase()
        self.admin_service = AdminService(self.db)
        self.api_base_url = api_base_url
        self.use_api = use_api
        self.llm = create_llm(temperature=0.1)

        # Build tools and agent
        self.tools = self._create_tools()
        self.agent = self._create_agent()

        logger.info(f"AdminAgent initialized (use_api={use_api})")

    def _create_tools(self) -> List:
        """Create the tools available to the admin agent.

        Returns:
            List of LangChain tools.
        """
        # Create tool functions that close over self
        agent_self = self

        @tool
        def submit_reservation(
            user_name: str,
            user_surname: str,
            car_number: str,
            parking_id: str,
            start_time: str,
            end_time: str,
        ) -> str:
            """Submit a new parking reservation request.

            Args:
                user_name: User's first name
                user_surname: User's last name
                car_number: Car registration number
                parking_id: ID of the parking space
                start_time: Start time in YYYY-MM-DD HH:MM format
                end_time: End time in YYYY-MM-DD HH:MM format

            Returns:
                Result message with reservation ID or error
            """
            return agent_self._submit_reservation(
                user_name, user_surname, car_number, parking_id, start_time, end_time
            )

        @tool
        def check_reservation_status(reservation_id: str) -> str:
            """Check the current status of a reservation by its ID.

            Args:
                reservation_id: The reservation ID to check

            Returns:
                Formatted status information
            """
            return agent_self._check_reservation_status(reservation_id)

        @tool
        def approve_reservation(
            reservation_id: str,
            admin_name: str,
            notes: str = "",
        ) -> str:
            """Approve a pending reservation. Only for admin use.

            Args:
                reservation_id: The reservation ID to approve
                admin_name: Name of the admin approving
                notes: Optional notes

            Returns:
                Success or error message
            """
            return agent_self._approve_reservation(
                reservation_id, admin_name, notes if notes else None
            )

        @tool
        def reject_reservation(
            reservation_id: str,
            admin_name: str,
            reason: str,
        ) -> str:
            """Reject a pending reservation with a reason. Only for admin use.

            Args:
                reservation_id: The reservation ID to reject
                admin_name: Name of the admin rejecting
                reason: Reason for rejection

            Returns:
                Success or error message
            """
            return agent_self._reject_reservation(reservation_id, admin_name, reason)

        @tool
        def list_pending_reservations() -> str:
            """Get a list of all pending reservations awaiting admin review.

            Returns:
                Formatted list of pending reservations
            """
            return agent_self._list_pending_reservations("")

        @tool
        def get_reservation_details(reservation_id: str) -> str:
            """Get full details of a specific reservation.

            Args:
                reservation_id: The reservation ID to get details for

            Returns:
                Full reservation details
            """
            return agent_self._get_reservation_details(reservation_id)

        return [
            submit_reservation,
            check_reservation_status,
            approve_reservation,
            reject_reservation,
            list_pending_reservations,
            get_reservation_details,
        ]

    def _create_agent(self):
        """Create the ReAct agent using LangGraph.

        Returns:
            Compiled LangGraph agent.
        """
        system_prompt = """You are an Admin Agent for a parking reservation system. Your job is to:
1. Submit new parking reservations on behalf of users
2. Check reservation status
3. Help administrators approve or reject reservations
4. List pending reservations

IMPORTANT RULES:
- When submitting a reservation, ensure all required fields are provided
- Always confirm actions with clear status messages
- For datetime, use format: YYYY-MM-DD HH:MM
- When checking status, provide clear information about the reservation state
"""
        return create_react_agent(
            self.llm,
            self.tools,
            prompt=system_prompt,
        )

    # ==================== TOOL IMPLEMENTATIONS ====================

    def _submit_reservation(
        self,
        user_name: str,
        user_surname: str,
        car_number: str,
        parking_id: str,
        start_time: str,
        end_time: str,
    ) -> str:
        """Submit a new reservation.

        This tool creates a reservation with 'pending' status and notifies admin.
        """
        try:
            # Parse datetime strings
            start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
            end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M")
        except ValueError as e:
            return f"Error: Invalid datetime format. Use YYYY-MM-DD HH:MM. Details: {e}"

        if self.use_api:
            return self._submit_via_api(
                user_name, user_surname, car_number, parking_id, start_dt, end_dt
            )
        else:
            return self._submit_via_db(
                user_name, user_surname, car_number, parking_id, start_dt, end_dt
            )

    def _submit_via_api(
        self,
        user_name: str,
        user_surname: str,
        car_number: str,
        parking_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> str:
        """Submit reservation via REST API."""
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.api_base_url}/api/reservations",
                    json={
                        "user_name": user_name,
                        "user_surname": user_surname,
                        "car_number": car_number,
                        "parking_id": parking_id,
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    return (
                        f"SUCCESS: Reservation {data['reservation_id']} created. "
                        f"Status: {data['status']}. Admin has been notified."
                    )
                else:
                    return f"ERROR: API returned {response.status_code}: {response.text}"

        except httpx.ConnectError:
            logger.warning("API not available, falling back to direct DB")
            return self._submit_via_db(
                user_name, user_surname, car_number, parking_id, start_time, end_time
            )
        except Exception as e:
            return f"ERROR: {str(e)}"

    def _submit_via_db(
        self,
        user_name: str,
        user_surname: str,
        car_number: str,
        parking_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> str:
        """Submit reservation directly to database."""
        import uuid

        res_id = f"RES_{uuid.uuid4().hex[:8].upper()}"

        success = self.db.create_reservation(
            res_id=res_id,
            user_name=user_name,
            user_surname=user_surname,
            car_number=car_number,
            parking_id=parking_id,
            start_time=start_time,
            end_time=end_time,
        )

        if success:
            logger.info(f"Reservation {res_id} created via AdminAgent")
            return (
                f"SUCCESS: Reservation {res_id} created with status 'pending'. "
                f"Awaiting admin approval."
            )
        else:
            return "ERROR: Failed to create reservation in database."

    def _check_reservation_status(self, reservation_id: str) -> str:
        """Check the status of a reservation."""
        if self.use_api:
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.get(
                        f"{self.api_base_url}/api/reservations/{reservation_id}"
                    )

                    if response.status_code == 200:
                        data = response.json()
                        return self._format_reservation_status(data)
                    elif response.status_code == 404:
                        return f"Reservation {reservation_id} not found."
                    else:
                        return f"Error: API returned {response.status_code}"

            except httpx.ConnectError:
                logger.warning("API not available, using direct DB")

        # Fallback to direct DB
        status = self.admin_service.get_reservation_status(reservation_id)
        if status:
            return self._format_reservation_status(status)
        return f"Reservation {reservation_id} not found."

    def _format_reservation_status(self, data: Dict) -> str:
        """Format reservation status for display."""
        status = data.get("status", "unknown").upper()
        res_id = data.get("id", "N/A")

        result = f"Reservation {res_id} - Status: {status}\n"
        result += f"- Name: {data.get('user_name', '')} {data.get('user_surname', '')}\n"
        result += f"- Car: {data.get('car_number', 'N/A')}\n"
        result += f"- Parking: {data.get('parking_name', data.get('parking_id', 'N/A'))}\n"

        if status == "CONFIRMED":
            result += f"- Approved by: {data.get('approved_by_admin', 'N/A')}\n"
        elif status == "REJECTED":
            result += f"- Rejected by: {data.get('approved_by_admin', 'N/A')}\n"
            result += f"- Reason: {data.get('rejection_reason', 'N/A')}\n"

        return result

    def _approve_reservation(
        self,
        reservation_id: str,
        admin_name: str,
        notes: Optional[str] = None,
    ) -> str:
        """Approve a reservation."""
        if self.use_api:
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(
                        f"{self.api_base_url}/api/admin/decision",
                        json={
                            "reservation_id": reservation_id,
                            "decision": "approve",
                            "admin_name": admin_name,
                            "notes": notes,
                        },
                    )

                    if response.status_code == 200:
                        data = response.json()
                        return f"SUCCESS: {data['message']}"
                    else:
                        return f"Error: API returned {response.status_code}: {response.text}"

            except httpx.ConnectError:
                logger.warning("API not available, using direct DB")

        # Fallback to direct DB
        result = self.admin_service.approve_reservation(reservation_id, admin_name, notes)
        if result["success"]:
            return f"SUCCESS: {result['message']}"
        return f"ERROR: {result['message']}"

    def _reject_reservation(
        self,
        reservation_id: str,
        admin_name: str,
        reason: str,
    ) -> str:
        """Reject a reservation."""
        if self.use_api:
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(
                        f"{self.api_base_url}/api/admin/decision",
                        json={
                            "reservation_id": reservation_id,
                            "decision": "reject",
                            "admin_name": admin_name,
                            "reason": reason,
                        },
                    )

                    if response.status_code == 200:
                        data = response.json()
                        return f"SUCCESS: {data['message']}"
                    else:
                        return f"Error: API returned {response.status_code}: {response.text}"

            except httpx.ConnectError:
                logger.warning("API not available, using direct DB")

        # Fallback to direct DB
        result = self.admin_service.reject_reservation(reservation_id, admin_name, reason)
        if result["success"]:
            return f"SUCCESS: {result['message']}"
        return f"ERROR: {result['message']}"

    def _list_pending_reservations(self, _: str = "") -> str:
        """List all pending reservations."""
        if self.use_api:
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.get(
                        f"{self.api_base_url}/api/reservations/pending/list"
                    )

                    if response.status_code == 200:
                        data = response.json()
                        return self._format_pending_list(data["reservations"])

            except httpx.ConnectError:
                logger.warning("API not available, using direct DB")

        # Fallback to direct DB
        pending = self.admin_service.get_pending_reservations()
        return self._format_pending_list(pending)

    def _format_pending_list(self, reservations: List[Dict]) -> str:
        """Format pending reservations list."""
        if not reservations:
            return "No pending reservations."

        result = f"Found {len(reservations)} pending reservation(s):\n\n"

        for res in reservations:
            result += f"* {res.get('id', 'N/A')}: "
            result += f"{res.get('user_name', '')} {res.get('user_surname', '')} "
            result += f"- {res.get('parking_id', 'N/A')}\n"

        return result

    def _get_reservation_details(self, reservation_id: str) -> str:
        """Get full details of a reservation."""
        return self._check_reservation_status(reservation_id.strip())

    # ==================== PUBLIC API ====================

    def submit_reservation(
        self,
        user_name: str,
        user_surname: str,
        car_number: str,
        parking_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """Submit a reservation (direct method, not via agent).

        Args:
            user_name: User's first name.
            user_surname: User's last name.
            car_number: Car registration number.
            parking_id: Parking space ID.
            start_time: Reservation start time.
            end_time: Reservation end time.

        Returns:
            Dict with success status and reservation ID.
        """
        result = self._submit_reservation(
            user_name=user_name,
            user_surname=user_surname,
            car_number=car_number,
            parking_id=parking_id,
            start_time=start_time.strftime("%Y-%m-%d %H:%M"),
            end_time=end_time.strftime("%Y-%m-%d %H:%M"),
        )

        # Parse result
        if "SUCCESS" in result:
            # Extract reservation ID from result
            import re
            match = re.search(r'RES_[A-Z0-9]+', result)
            res_id = match.group(0) if match else None

            return {
                "success": True,
                "reservation_id": res_id,
                "message": result,
            }
        else:
            return {
                "success": False,
                "reservation_id": None,
                "message": result,
            }

    def run(self, query: str) -> str:
        """Run the agent with a query.

        Args:
            query: The query or instruction for the agent.

        Returns:
            Agent's response.
        """
        try:
            result = self.agent.invoke({"messages": [("user", query)]})
            # Extract the final message from the agent
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, "content"):
                    return last_message.content
                return str(last_message)
            return "No response generated."
        except Exception as e:
            logger.error(f"AdminAgent error: {e}")
            return f"Error processing request: {e}"

    def close(self):
        """Close database connections."""
        self.db.close()
