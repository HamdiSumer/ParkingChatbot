#!/usr/bin/env python3
"""Quick test to verify chatbot initialization."""
import sys
from src.app import create_app

try:
    print("Initializing chatbot...")
    app = create_app(skip_milvus=False)
    print("✓ Chatbot initialized successfully!")
    print(f"  - Vector store: {app.vector_store is not None}")
    print(f"  - RAG retriever: {app.rag_retriever is not None}")
    print(f"  - Workflow: {app.workflow is not None}")

    if app.workflow:
        print("\n✓ All components ready!")
        print("\nTesting with simple message...")
        result = app.process_user_message("Hello")
        print(f"Response: {result.get('response')}")
    else:
        print("\n✗ Workflow not initialized")
        sys.exit(1)

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
