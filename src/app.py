"""Main application for the parking chatbot."""
from src.config import get_config
from src.utils.logging import logger, setup_logging
from src.rag.embeddings import create_embeddings
from src.rag.retriever import ParkingRAGRetriever
from src.database.milvus_db import create_milvus_connection, ingest_static_data
from src.database.sql_db import ParkingDatabase
from src.agents.workflow import ParkingChatbotWorkflow
from src.guardrails.filter import DataProtectionFilter
from src.evaluation.test_data import get_sample_parking_documents
from langchain_core.documents import Document
from typing import Optional


class ParkingChatbotApp:
    """Main application class for the parking chatbot."""

    def __init__(self, skip_milvus: bool = False):
        """Initialize the parking chatbot application.

        Args:
            skip_milvus: If True, skip Milvus initialization (useful for testing without Milvus).
        """
        setup_logging()
        logger.info("Initializing Parking Chatbot Application...")

        self.config = get_config()
        self.skip_milvus = skip_milvus

        # Initialize components
        self.embeddings = create_embeddings()
        logger.info("✓ Embeddings initialized")

        # Initialize databases
        self.db = ParkingDatabase()
        logger.info("✓ SQL Database initialized")

        # Initialize Vector Database (Weaviate, Pinecone, or Milvus)
        self.vector_store = None
        if not skip_milvus:
            try:
                provider = self.config.VECTOR_DB_PROVIDER
                if provider == "weaviate":
                    from src.database.weaviate_db import create_weaviate_connection
                    self.vector_store = create_weaviate_connection(self.embeddings)
                    logger.info("✓ Weaviate Vector Database initialized")
                elif provider == "pinecone":
                    from src.database.pinecone_db import create_pinecone_connection
                    self.vector_store = create_pinecone_connection(self.embeddings)
                    logger.info("✓ Pinecone Vector Database initialized")
                else:  # milvus
                    self.vector_store = create_milvus_connection(self.embeddings)
                    logger.info("✓ Milvus Vector Database initialized")
            except Exception as e:
                logger.warning(f"Vector DB initialization failed: {e}. Continuing without vector DB.")

        # Initialize RAG retriever
        if self.vector_store:
            self.rag_retriever = ParkingRAGRetriever(self.vector_store)
            logger.info("✓ RAG Retriever initialized")
        else:
            self.rag_retriever = None
            logger.warning("RAG Retriever skipped (no vector store)")

        # Initialize guard rails
        self.guard_rails = DataProtectionFilter()
        logger.info("✓ Guard Rails initialized")

        # Initialize workflow
        if self.rag_retriever:
            self.workflow = ParkingChatbotWorkflow(
                self.rag_retriever, self.db, self.guard_rails
            )
            logger.info("✓ Chatbot Workflow initialized")
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

    def shutdown(self):
        """Cleanup and shutdown the application."""
        logger.info("Shutting down application...")
        if self.db:
            logger.info("Database closed")
        logger.info("Application shutdown complete")


def create_app(skip_milvus: bool = False) -> ParkingChatbotApp:
    """Factory function to create the application.

    Args:
        skip_milvus: If True, skip Milvus initialization.

    Returns:
        ParkingChatbotApp instance.
    """
    return ParkingChatbotApp(skip_milvus=skip_milvus)
