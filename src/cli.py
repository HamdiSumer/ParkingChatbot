"""Command-line interface for the parking chatbot."""
import sys
from src.app import create_app
from src.evaluation.runner import EvaluationRunner
from src.utils.logging import logger


def print_welcome():
    """Print welcome message."""
    print("\n" + "=" * 80)
    print("PARKING CHATBOT - Powered by Ollama & LangChain".center(80))
    print("=" * 80)
    print("Type 'help' for available commands, 'quit' to exit\n")


def print_help():
    """Print help information."""
    print("\nAvailable Commands:")
    print("  quit/exit         - Exit the chatbot")
    print("  help              - Show this help message")
    print("  parking list      - List all parking spaces")
    print("  parking info <id> - Get info about a parking space")
    print("  evaluate          - Run system evaluation tests")
    print("  clear             - Clear screen\n")


def format_response(result: dict) -> str:
    """Format chatbot response for display.

    Args:
        result: Response dictionary from chatbot.

    Returns:
        Formatted string for display.
    """
    response_text = result.get("response", "No response")

    if result.get("safety_issue"):
        response_text = f"⚠️  BLOCKED: {response_text}"

    sources = result.get("sources", [])
    if sources:
        response_text += "\n\nSources:"
        for i, source in enumerate(sources, 1):
            response_text += f"\n  [{i}] {source.page_content[:100]}..."

    return response_text


def main():
    """Main CLI loop."""
    print_welcome()

    # Initialize application
    try:
        app = create_app(skip_vector_db=False)
        app.ingest_sample_data()
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        print(f"Error: Failed to initialize chatbot: {e}")
        print("Trying to continue with limited functionality...")
        try:
            app = create_app(skip_vector_db=True)
        except Exception as e2:
            print(f"Fatal error: {e2}")
            sys.exit(1)

    print("Chatbot ready! Type 'help' for available commands.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["quit", "exit"]:
                print("\nGoodbye!")
                break

            elif user_input.lower() == "help":
                print_help()

            elif user_input.lower() == "clear":
                import os
                os.system("clear" if os.name == "posix" else "cls")
                print_welcome()

            elif user_input.lower() == "parking list":
                spaces = app.list_parking_spaces()
                print("\nAvailable Parking Spaces:")
                for space in spaces:
                    print(f"  • {space['name']} ({space['id']})")
                    print(f"    Location: {space['location']}")
                    print(f"    Capacity: {space['capacity']} | Available: {space['available']}")
                    print(f"    Price: ${space['price_per_hour']:.2f}/hour\n")

            elif user_input.lower().startswith("parking info"):
                parts = user_input.split()
                if len(parts) >= 3:
                    parking_id = parts[2]
                    info = app.get_parking_info(parking_id)
                    if info:
                        print(f"\n{info['name']} ({parking_id})")
                        print(f"  Location: {info['location']}")
                        print(f"  Capacity: {info['capacity']}")
                        print(f"  Available: {info['available_spaces']}")
                        print(f"  Price: ${info['price_per_hour']:.2f}/hour")
                        print(f"  Status: {'Open' if info['is_open'] else 'Closed'}\n")
                    else:
                        print(f"\nParking space '{parking_id}' not found.\n")
                else:
                    print("\nUsage: parking info <parking_id>\n")

            elif user_input.lower() == "evaluate":
                print("\nStarting evaluation tests...")
                print("Note: This requires Ollama and Milvus to be running.\n")

                if not app.rag_retriever or not app.workflow:
                    print("Evaluation skipped: Required components not initialized\n")
                    continue

                try:
                    evaluator = EvaluationRunner()
                    sample_queries = [
                        "Where is downtown parking?",
                        "What are the parking prices?",
                        "How do I make a reservation?",
                    ]

                    evaluator.run_full_evaluation(
                        app.rag_retriever, app.workflow, app.db, sample_queries
                    )

                    # Save reports
                    evaluator.report.save_report("./reports/evaluation_report.md")
                    evaluator.report.save_json_results("./reports/evaluation_results.json")

                    print("\n✓ Evaluation complete!")
                    print("  Report saved to: ./reports/evaluation_report.md")
                    print("  Results saved to: ./reports/evaluation_results.json\n")

                except Exception as e:
                    logger.error(f"Evaluation failed: {e}")
                    print(f"Error during evaluation: {e}\n")

            else:
                # Process regular message
                result = app.process_user_message(user_input)

                if result.get("error"):
                    print(f"\nBot: ⚠️  {result.get('response')}\n")
                else:
                    formatted = format_response(result)
                    print(f"\nBot: {formatted}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error in CLI: {e}")
            print(f"Error: {e}\n")

    app.shutdown()


if __name__ == "__main__":
    main()
