"""Configuration management for the parking chatbot."""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration."""

    # ========== LLM Provider Configuration (Required) ==========
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()  # ollama, openai, gemini, anthropic

    # Ollama
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")

    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

    # Google Gemini
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")

    # Anthropic Claude
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")

    # ========== Vector Database Configuration (Required) ==========
    VECTOR_DB_PROVIDER = os.getenv("VECTOR_DB_PROVIDER", "weaviate").lower()  # weaviate, pinecone, or milvus

    # Weaviate (Open source, Docker)
    WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "http://localhost:8080")
    WEAVIATE_INDEX_NAME = os.getenv("WEAVIATE_INDEX_NAME", "ParkingStaticData")

    # Milvus (Local, Docker - complex setup)
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "parking_static_data")

    # Pinecone (Cloud, proprietary)
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "parking-static-data")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

    # ========== SQLite Configuration ==========
    SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/parking.db")

    # ========== Vector Embedding Configuration ==========
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

    # ========== Application Configuration ==========
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"


def get_config() -> Config:
    """Get application configuration."""
    return Config()
