"""Main application for the parking chatbot."""
from src.config import get_config
from src.utils.logging import logger, setup_logging
from src.rag.embeddings import create_embeddings
from src.rag.retriever import ParkingRAGRetriever
from src.rag.sql_agent import create_sql_agent
from src.rag.llm_provider import create_llm
from src.database.weaviate_db import create_weaviate_connection
from src.database.sql_db import ParkingDatabase
from src.agents.workflow import ParkingChatbotWorkflow
from src.guardrails.filter import DataProtectionFilter
from src.evaluation.test_data import get_sample_parking_documents
from src.admin.admin_service import AdminService
from typing import Optional


class ParkingChatbotApp:
    """Main application class for the parking chatbot."""

    def __init__(self, skip_vector_db: bool = False):
        """Initialize the parking chatbot application.

        Args:
            skip_vector_db: If True, skip Weaviate initialization (useful for testing without vector DB).
        """
        setup_logging()
        logger.info("Initializing Parking Chatbot Application...")

        self.config = get_config()
        self.skip_vector_db = skip_vector_db

        # Initialize components
        self.embeddings = create_embeddings()
        logger.info("✓ Embeddings initialized")

        # Initialize shared LLM (used by RAG retriever and SQL agent)
        self.llm = create_llm(temperature=0.3)
        logger.info("✓ LLM initialized (shared instance)")

        # Initialize databases
        self.db = ParkingDatabase()
        logger.info("✓ SQL Database initialized")

        # Initialize admin service
        self.admin_service = AdminService(self.db)
        logger.info("✓ Admin Service initialized")

        # Initialize Vector Database (Weaviate only)
        self.vector_store = None
        if not self.skip_vector_db:
            try:
                self.vector_store = create_weaviate_connection(self.embeddings)
                logger.info("✓ Weaviate Vector Database initialized")
            except Exception as e:
                logger.warning(f"Vector DB initialization failed: {e}. Continuing without vector DB.")

        # Initialize SQL Agent for hybrid retrieval (reusing shared LLM)
        self.sql_agent = None
        if self.db:
            try:
                self.sql_agent = create_sql_agent(self.db, llm=self.llm)
                if self.sql_agent:
                    logger.info("✓ SQL Agent initialized for hybrid retrieval")
                else:
                    logger.warning("SQL Agent initialization failed")
            except Exception as e:
                logger.warning(f"Could not initialize SQL Agent: {e}")

        # Initialize RAG retriever with hybrid retrieval (Vector DB + SQL Agent)
        if self.vector_store:
            self.rag_retriever = ParkingRAGRetriever(
                self.vector_store,
                llm=self.llm,  # Reuse shared LLM
                db=self.db,
                sql_agent=self.sql_agent
            )
            logger.info("✓ RAG Retriever initialized with hybrid retrieval (Vector DB + SQL Agent)")
        else:
            self.rag_retriever = None
            logger.warning("RAG Retriever skipped (no vector store)")

        # Initialize guard rails
        self.guard_rails = DataProtectionFilter()
        logger.info("✓ Guard Rails initialized")

        # Initialize workflow with ReAct agent
        if self.rag_retriever:
            self.workflow = ParkingChatbotWorkflow(
                self.rag_retriever, self.db, self.guard_rails,
                sql_agent=self.sql_agent  # Pass SQL agent for intelligent tool selection
            )
            logger.info("✓ Chatbot Workflow initialized with ReAct agent")
        else:
            self.workflow = None
            logger.warning("Chatbot Workflow skipped (no RAG retriever)")

        logger.info("Application initialization complete!")

    def ingest_sample_data(self):
        """Ingest sample parking data into the system."""
        if not self.vector_store:
            logger.warning("Cannot ingest data: Vector store not initialized")
            return

        try:
            documents = get_sample_parking_documents()

            # Use add_documents method which works for both Weaviate and Milvus
            try:
                self.vector_store.add_documents(documents)
                logger.info(f"Ingested {len(documents)} documents into vector store")
            except Exception as e:
                # Fallback to old ingest_static_data for backward compatibility
                logger.warning(f"add_documents failed, trying fallback: {e}")
                ingest_static_data(self.vector_store, documents)

            # Add sample parking spaces to SQL DB
            self._add_sample_parking_spaces()

            logger.info("Sample data ingested successfully")
        except Exception as e:
            logger.error(f"Failed to ingest sample data: {e}")

    def _add_sample_parking_spaces(self):
        """Add sample parking spaces to the database."""
        parking_spaces = [
            {
                "id": "downtown_1",
                "name": "Downtown Parking Garage",
                "location": "123 Main Street",
                "capacity": 500,
                "price_per_hour": 5.0,
            },
            {
                "id": "airport_1",
                "name": "Airport Parking",
                "location": "2 miles from terminal",
                "capacity": 1000,
                "price_per_hour": 3.0,
            },
            {
                "id": "riverside_1",
                "name": "Riverside Parking Lot",
                "location": "Near the river",
                "capacity": 200,
                "price_per_hour": 4.0,
            },
        ]

        for space in parking_spaces:
            self.db.add_parking_space(
                id=space["id"],
                name=space["name"],
                location=space["location"],
                capacity=space["capacity"],
                price_per_hour=space["price_per_hour"],
            )

        logger.info(f"Added {len(parking_spaces)} sample parking spaces")

    def process_user_message(self, message: str) -> dict:
        """Process a user message through the chatbot.

        Args:
            message: User message to process.

        Returns:
            Dictionary with chatbot response and metadata.
        """
        if not self.workflow:
            logger.error("Workflow not initialized")
            return {"response": "Error: Chatbot not properly initialized", "error": True}

        try:
            result = self.workflow.invoke(message)
            logger.info(f"Message processed successfully")
            return result
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {"response": f"Error processing your request: {e}", "error": True}

    def get_parking_info(self, parking_id: str) -> Optional[dict]:
        """Get information about a specific parking space.

        Args:
            parking_id: ID of the parking space.

        Returns:
            Dictionary with parking information or None.
        """
        return self.db.get_parking_space(parking_id)

    def list_parking_spaces(self) -> list:
        """Get list of all parking spaces.

        Returns:
            List of parking space information.
        """
        # Query directly from database session
        session = self.db.get_session()
        try:
            from src.database.sql_db import ParkingSpace
            spaces = session.query(ParkingSpace).all()
            return [
                {
                    "id": s.id,
                    "name": s.name,
                    "location": s.location,
                    "capacity": s.capacity,
                    "available": s.available_spaces,
                    "price_per_hour": s.price_per_hour,
                }
                for s in spaces
            ]
        finally:
            session.close()

    # ==================== ADMIN METHODS ====================

    def get_pending_reservations(self) -> list:
        """Get all pending reservations awaiting admin review.

        Returns:
            List of pending reservation dictionaries.
        """
        return self.admin_service.get_pending_reservations()

    def approve_reservation(self, res_id: str, admin_name: str, notes: str = None) -> dict:
        """Approve a reservation.

        Args:
            res_id: Reservation ID to approve.
            admin_name: Name of admin performing approval.
            notes: Optional notes about the approval.

        Returns:
            Dict with success status and message.
        """
        return self.admin_service.approve_reservation(res_id, admin_name, notes)

    def reject_reservation(self, res_id: str, admin_name: str, reason: str) -> dict:
        """Reject a reservation.

        Args:
            res_id: Reservation ID to reject.
            admin_name: Name of admin performing rejection.
            reason: Reason for rejection.

        Returns:
            Dict with success status and message.
        """
        return self.admin_service.reject_reservation(res_id, admin_name, reason)

    def check_reservation_status(self, res_id: str) -> Optional[dict]:
        """Check the status of a reservation.

        Args:
            res_id: Reservation ID to check.

        Returns:
            Dict with status info or None if not found.
        """
        return self.admin_service.get_reservation_status(res_id)

    def get_reservation_details(self, res_id: str) -> Optional[dict]:
        """Get full details of a reservation.

        Args:
            res_id: Reservation ID.

        Returns:
            Reservation details dict or None if not found.
        """
        return self.admin_service.get_reservation_details(res_id)

    def shutdown(self):
        """Cleanup and shutdown the application."""
        logger.info("Shutting down application...")
        if self.sql_agent and hasattr(self.sql_agent, 'close'):
            self.sql_agent.close()
        if self.db:
            self.db.close()
        logger.info("Application shutdown complete")


def create_app(skip_vector_db: bool = False) -> ParkingChatbotApp:
    """Factory function to create the application.

    Args:
        skip_vector_db: If True, skip Weaviate initialization.

    Returns:
        ParkingChatbotApp instance.
    """
    return ParkingChatbotApp(skip_vector_db=skip_vector_db)
