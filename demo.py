#!/usr/bin/env python3
"""Demo script showing parking chatbot capabilities."""

from src.app import create_app
from src.evaluation.runner import EvaluationRunner
from src.evaluation.test_data import get_safety_test_cases
import time


def demo_information_queries():
    """Demo: Information retrieval using RAG."""
    print("\n" + "=" * 80)
    print("DEMO 1: Information Retrieval (RAG)".center(80))
    print("=" * 80 + "\n")

    app = create_app(skip_milvus=True)  # Can work without Milvus for demo
    app.ingest_sample_data()

    queries = [
        "Where is downtown parking located?",
        "What are the parking prices?",
        "How do I make a reservation?",
        "What are the airport parking hours?",
    ]

    for query in queries:
        print(f"Q: {query}")
        result = app.process_user_message(query)
        print(f"A: {result['response']}\n")
        time.sleep(1)


def demo_reservation_collection():
    """Demo: Reservation data collection."""
    print("\n" + "=" * 80)
    print("DEMO 2: Reservation Collection".center(80))
    print("=" * 80 + "\n")

    app = create_app(skip_milvus=True)
    app.ingest_sample_data()

    print("Simulating reservation process...\n")

    messages = [
        "I want to book a parking space",
        "John",
        "Doe",
        "ABC-123",
        "downtown_1",
        "2024-03-01 09:00",
        "2024-03-01 17:00",
    ]

    for msg in messages:
        print(f"User: {msg}")
        result = app.process_user_message(msg)
        print(f"Bot: {result['response']}\n")
        time.sleep(0.5)


def demo_safety_filtering():
    """Demo: Safety and guard rails."""
    print("\n" + "=" * 80)
    print("DEMO 3: Safety & Guard Rails".center(80))
    print("=" * 80 + "\n")

    from src.guardrails.filter import DataProtectionFilter

    guard_rails = DataProtectionFilter()
    test_cases = get_safety_test_cases()

    print(f"Testing {len(test_cases)} messages for safety...\n")

    safe_count = 0
    blocked_count = 0

    for message, should_block in test_cases[:5]:  # Show first 5
        is_safe, reason = guard_rails.check_safety(message)

        status = "✓ ALLOWED" if is_safe else "✗ BLOCKED"
        print(f"{status}: '{message[:50]}...'")
        if not is_safe:
            print(f"   Reason: {reason}")
            blocked_count += 1
        else:
            safe_count += 1
        print()

    print(f"Summary: {safe_count} allowed, {blocked_count} blocked")


def demo_parking_info():
    """Demo: Parking information management."""
    print("\n" + "=" * 80)
    print("DEMO 4: Parking Space Management".center(80))
    print("=" * 80 + "\n")

    app = create_app(skip_milvus=True)
    app.ingest_sample_data()

    spaces = app.list_parking_spaces()

    print(f"Available Parking Spaces ({len(spaces)} total):\n")

    for space in spaces:
        available_pct = (space["available"] / space["capacity"]) * 100
        availability_bar = "█" * int(available_pct / 10) + "░" * (10 - int(available_pct / 10))

        print(f"📍 {space['name']}")
        print(f"   ID: {space['id']}")
        print(f"   Location: {space['location']}")
        print(f"   Availability: {space['available']}/{space['capacity']} " + availability_bar)
        print(f"   Price: ${space['price_per_hour']:.2f}/hour\n")


def demo_evaluation():
    """Demo: System evaluation."""
    print("\n" + "=" * 80)
    print("DEMO 5: System Evaluation".center(80))
    print("=" * 80 + "\n")

    try:
        app = create_app(skip_milvus=True)
        app.ingest_sample_data()

        if not app.rag_retriever or not app.workflow:
            print("Skipping evaluation: RAG components not available")
            print("(Requires Ollama and Milvus to be running for full evaluation)")
            return

        print("Running evaluation tests...\n")

        evaluator = EvaluationRunner()
        sample_queries = [
            "Where is downtown parking?",
            "What are the prices?",
            "How do I make a reservation?",
        ]

        start_time = time.time()

        # Run evaluations
        evaluator.evaluate_safety_system()
        print("✓ Safety evaluation complete")

        evaluator.evaluate_reservation_process(app.db)
        print("✓ Reservation evaluation complete")

        if app.rag_retriever:
            evaluator.evaluate_rag_system(app.rag_retriever)
            print("✓ RAG evaluation complete")

        if app.workflow:
            evaluator.evaluate_performance(app.workflow, sample_queries)
            print("✓ Performance evaluation complete")

        elapsed = time.time() - start_time

        # Save reports
        evaluator.report.save_report("./reports/demo_evaluation_report.md")
        evaluator.report.save_json_results("./reports/demo_evaluation_results.json")

        print(f"\n✓ Evaluation completed in {elapsed:.2f} seconds")
        print("✓ Report saved to: ./reports/demo_evaluation_report.md")
        print("✓ Results saved to: ./reports/demo_evaluation_results.json")

    except Exception as e:
        print(f"Evaluation failed: {e}")
        print("(Ensure Ollama and optionally Milvus are running)")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + "PARKING CHATBOT - COMPREHENSIVE DEMO".center(78) + "║")
    print("╚" + "=" * 78 + "╝")

    try:
        demo_information_queries()
        demo_safety_filtering()
        demo_parking_info()
        demo_reservation_collection()
        demo_evaluation()

        print("\n" + "=" * 80)
        print("DEMO COMPLETE".center(80))
        print("=" * 80 + "\n")
        print("Next steps:")
        print("  1. Run the interactive chatbot: python main.py")
        print("  2. Try different queries and reservation processes")
        print("  3. Review the evaluation reports in ./reports/")
        print()

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo error: {e}")
        print("Make sure Ollama is running: ollama serve")


if __name__ == "__main__":
    main()
