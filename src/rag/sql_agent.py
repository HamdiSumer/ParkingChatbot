"""SQL Agent for hybrid retrieval - intelligently queries the SQL database."""
from typing import Any, Optional
from src.utils.logging import logger
from src.rag.llm_provider import create_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def create_sql_agent(db, llm: Optional[Any] = None):
    """Create a SQL agent that can inspect schema and execute queries.

    Uses a chain-based approach (works with any LLM including Ollama):
    1. Inspects the database schema
    2. Uses LLM to decide what queries would help answer a question
    3. Executes those queries dynamically
    4. Returns the results

    Args:
        db: ParkingDatabase instance (SQLite).
        llm: Optional LLM instance to reuse. If None, creates a new one.

    Returns:
        SQL agent executor that can be invoked with user questions.
    """
    try:
        from langchain_community.utilities import SQLDatabase

        # Create SQLDatabase wrapper from the SQL DB connection
        # This inspects the schema automatically
        sql_db = SQLDatabase.from_uri(
            f"sqlite:///{db.db_path}",
            schema="main",
            view_support=True
        )

        # Get the database schema
        db_schema = sql_db.get_table_info()
        logger.info("SQL Database schema loaded")
        logger.info(f"Available tables: {sql_db.get_usable_table_names()}")

        # Use provided LLM or create new one
        if llm is None:
            llm = create_llm(temperature=0.0)  # 0 temperature for precise SQL
            logger.info("Created new LLM for SQL Agent")
        else:
            logger.info("Reusing provided LLM for SQL Agent")

        # Create a simple wrapper for the SQL agent
        class SQLAgentExecutor:
            """Chain-based SQL agent executor."""

            def __init__(self, sql_db, llm, schema):
                self.sql_db = sql_db
                self.llm = llm
                self.schema = schema

                # Create prompt for deciding what queries to run
                self.decision_prompt = PromptTemplate(
                    template="""Given the following database schema, generate SQL queries to answer the user's question.

Database Schema:
{schema}

User Question: {question}

Instructions:
1. Generate ONE SQL SELECT query that helps answer the question
2. Do NOT include explanations or comments
3. Output ONLY the SQL query on a single line
4. If no query is helpful, output: SKIP

Example outputs:
- SELECT * FROM parking_spaces WHERE id='downtown_1'
- SELECT COUNT(*) as count FROM parking_spaces WHERE is_open=1
- SKIP

Now generate your SQL query:""",
                    input_variables=["schema", "question"]
                )
                self.decision_chain = self.decision_prompt | llm | StrOutputParser()

            def invoke(self, input_dict: dict) -> dict:
                """Invoke the SQL agent.

                Args:
                    input_dict: Dictionary with "input" key containing the user's question.

                Returns:
                    Dictionary with "output" key containing the agent's response.
                """
                try:
                    question = input_dict.get("input", "")
                    logger.info(f"SQL Agent processing: {question[:50]}...")

                    # Step 1: Use LLM to decide what queries to run
                    sql_queries_text = self.decision_chain.invoke({
                        "schema": self.schema,
                        "question": question
                    })

                    logger.info(f"Generated SQL queries: {sql_queries_text[:100]}...")

                    # Step 2: Execute the queries
                    results = []
                    if sql_queries_text.strip().upper() != "SKIP":
                        # Clean up the generated SQL
                        queries = [q.strip() for q in sql_queries_text.strip().split('\n') if q.strip() and q.strip().upper() != "SKIP"]

                        for query in queries:
                            # Skip non-SELECT queries (basic safety)
                            query_upper = query.upper()
                            if not query_upper.startswith("SELECT"):
                                logger.warning(f"Skipping non-SELECT query: {query}")
                                continue

                            # Sanitize query (basic protection)
                            if any(dangerous in query_upper for dangerous in ["DROP", "DELETE", "INSERT", "UPDATE", "TRUNCATE", ";DROP", ";DELETE"]):
                                logger.warning(f"Blocked dangerous query: {query}")
                                continue

                            try:
                                # Execute the query
                                result = self.sql_db.run(query)
                                if result:
                                    results.append(f"Query: {query}\nResult: {result}")
                                    logger.info(f"Query executed successfully")
                                else:
                                    logger.info(f"Query returned no results: {query}")
                            except Exception as e:
                                logger.warning(f"Query execution failed ({query}): {e}")

                    # Step 3: Format the results
                    if results:
                        output = "Current Database Information:\n" + "\n\n".join(results)
                    else:
                        output = ""

                    logger.info(f"SQL Agent result: {output[:100]}...")
                    return {"output": output}

                except Exception as e:
                    logger.error(f"SQL Agent error: {e}")
                    return {"output": ""}

            def close(self):
                """Close the SQL database connection."""
                try:
                    if hasattr(self.sql_db, '_engine'):
                        self.sql_db._engine.dispose()
                        logger.info("SQL Agent database connection closed")
                except Exception as e:
                    logger.error(f"Error closing SQL Agent connection: {e}")

        return SQLAgentExecutor(sql_db, llm, db_schema)

    except ImportError as e:
        logger.warning(f"SQL Agent dependencies not available: {e}. Hybrid retrieval will be limited.")
        return None
    except Exception as e:
        logger.error(f"Failed to create SQL agent: {e}")
        return None
