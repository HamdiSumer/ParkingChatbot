"""Tool definitions for the ReAct agent."""
from typing import Any, List, Optional
from langchain_core.documents import Document
from src.utils.logging import logger


class VectorSearchTool:
    """Tool for searching static parking information from vector database."""

    name: str = "vector_search"
    description: str = """Search the parking knowledge base for static information.
Use this for: parking locations, addresses, policies, rules, booking process, general information, FAQs.
Examples: "Where is downtown parking?", "What are the parking rules?", "How do I make a reservation?" """

    def __init__(self, vector_store):
        """Initialize with vector store.

        Args:
            vector_store: Weaviate vector store instance.
        """
        self.vector_store = vector_store

    def invoke(self, query: str, k: int = 3) -> dict:
        """Search the vector database.

        Args:
            query: Search query.
            k: Number of documents to retrieve.

        Returns:
            Dict with 'documents' list and 'formatted' string.
        """
        try:
            documents = self.vector_store.as_retriever(
                search_kwargs={"k": k}
            ).invoke(query)

            formatted = self._format_results(documents)
            logger.info(f"Vector search returned {len(documents)} documents")

            return {
                "documents": documents,
                "formatted": formatted,
                "success": True
            }
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return {
                "documents": [],
                "formatted": "",
                "success": False,
                "error": str(e)
            }

    def _format_results(self, documents: List[Document]) -> str:
        """Format documents for context."""
        if not documents:
            return "No relevant information found."

        parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.page_content.strip()
            source = doc.metadata.get("source", "parking_info")
            parts.append(f"[{i}] {content}")

        return "\n\n".join(parts)


class SQLQueryTool:
    """Tool for querying real-time parking data from SQL database."""

    name: str = "sql_query"
    description: str = """Query real-time parking data from the database.
Use this for: current availability, real-time prices, specific parking space status, how many spaces.
Examples: "How many spaces available?", "Current price at airport?", "Is downtown parking open?" """

    def __init__(self, sql_agent):
        """Initialize with SQL agent.

        Args:
            sql_agent: SQL agent executor instance.
        """
        self.sql_agent = sql_agent

    def invoke(self, query: str) -> dict:
        """Query the SQL database.

        Args:
            query: Natural language query about parking data.

        Returns:
            Dict with 'result' string and metadata.
        """
        if not self.sql_agent:
            return {
                "result": "",
                "success": False,
                "error": "SQL agent not available"
            }

        try:
            result = self.sql_agent.invoke({"input": query})
            output = result.get("output", "")

            logger.info(f"SQL query executed successfully")

            return {
                "result": output,
                "formatted": output if output else "No data found.",
                "success": bool(output)
            }
        except Exception as e:
            logger.error(f"SQL query failed: {e}")
            return {
                "result": "",
                "formatted": "",
                "success": False,
                "error": str(e)
            }


class ToolRegistry:
    """Registry of available tools for the agent."""

    def __init__(self, vector_store=None, sql_agent=None):
        """Initialize tool registry.

        Args:
            vector_store: Weaviate vector store (optional).
            sql_agent: SQL agent executor (optional).
        """
        self.tools = {}

        if vector_store:
            self.tools["vector_search"] = VectorSearchTool(vector_store)

        if sql_agent:
            self.tools["sql_query"] = SQLQueryTool(sql_agent)

    def get_tool(self, name: str) -> Optional[Any]:
        """Get a tool by name."""
        return self.tools.get(name)

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())

    def get_tools_description(self) -> str:
        """Get formatted description of all available tools."""
        descriptions = []

        for name, tool in self.tools.items():
            descriptions.append(f"- {name}: {tool.description}")

        # Always include direct_response and start_reservation
        descriptions.append(
            "- direct_response: Respond directly without using any tools. "
            "Use for: greetings, thanks, chitchat, simple questions that don't need data lookup."
        )
        descriptions.append(
            "- start_reservation: Begin the parking reservation process. "
            "Use when: user wants to book, reserve, or make a parking reservation."
        )
        descriptions.append(
            "- synthesize: Generate final response from gathered information. "
            "Use when: you have enough information to answer the user's question."
        )

        return "\n".join(descriptions)

    def invoke_tool(self, name: str, query: str) -> dict:
        """Invoke a tool by name.

        Args:
            name: Tool name.
            query: Query to pass to the tool.

        Returns:
            Tool result dict.
        """
        tool = self.get_tool(name)
        if not tool:
            return {"success": False, "error": f"Tool '{name}' not found"}

        return tool.invoke(query)
