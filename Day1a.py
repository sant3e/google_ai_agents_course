# -*- coding: utf-8 -*-
"""
Day 1A: A Basic "Hello, World!" Agent with Google's ADK.

This script serves as a foundational, educational example for building an AI agent
using Google's Agent Development Kit (ADK). It demonstrates the core components
and architectural patterns required to create a simple, functional agent.

Key Concepts Illustrated:
- Environment Configuration: Securely loading API keys.
- Agent Definition: Structuring an agent with instructions, a model, and tools.
- Agent Execution: Using a runner to process a user query.
- Asynchronous Operation: Maintaining the core async pattern for agent interaction.

Prerequisites:
- google-adk
- python-dotenv
- A .env file in the root directory with a GOOGLE_API_KEY.
"""

from initial_setup import GEMINI_MODEL, setup_environment
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search

# --- Constants and Configuration ---


def create_search_assistant_agent() -> Agent:
    """
    Defines and constructs the primary agent for this application.

    This function encapsulates the agent's entire definition, including its
    identity, capabilities, and instructions. This modular approach makes it
    easy to configure, test, and reuse the agent's definition.

    Returns:
        Agent: An instance of a configured ADK Agent.
    """
    assistant_instruction = """
    You are a helpful and friendly assistant.

    Your primary goal is to provide accurate and timely information.
    1. If a question seems to require up-to-date information or knowledge
       of current events, you MUST use the `google_search` tool.
    2. For general knowledge questions within your training data, you can
       answer directly.
    3. Always provide clear, concise, and helpful responses.
    """

    search_assistant = Agent(
        name="search_assistant",
        model=GEMINI_MODEL,
        description="An assistant agent equipped with Google Search.",
        instruction=assistant_instruction,
        tools=[google_search],
    )
    print(f"🤖 Agent '{search_assistant.name}' created successfully.")
    return search_assistant


async def main() -> None:
    """
    The main asynchronous entry point for running the agent.

    This function orchestrates the entire process: setting up the environment,
    creating the agent, initializing the runner, and executing a sample query.
    Using an async main function is best practice when working with async
    libraries like the ADK.
    """
    print("---" + " Starting Basic Search Agent ---")
    setup_environment()
    root_agent = create_search_assistant_agent()

    # The InMemoryRunner is a simple way to run and test an agent locally.
    # It handles the state and execution flow within the script's memory.
    runner = InMemoryRunner(agent=root_agent)
    print("🏃 Runner initialized. Ready to process queries.")

    user_query = "What is the current status of the Artemis program?"

    print(f"\n💬 User Query: '{user_query}'")
    print("🧠 Agent is thinking...")

    # The `run_debug` method processes the query and provides detailed
    # step-by-step output of the agent's reasoning and tool usage.
    response = await runner.run_debug(user_query)

    print("\n✅ Agent execution complete.")
    print("--- Agent's Final Response ---")
    print(response)
    print("------------------------------")


if __name__ == "__main__":
    # This block handles running the async main function in different environments.
    # `asyncio.run()` is the standard for scripts, but it fails in environments
    # with a running event loop (like Jupyter).
    try:
        import asyncio
        asyncio.run(main())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            # If in a Jupyter notebook or other interactive environment, schedule
            # the main function as a task on the existing loop.
            import asyncio
            print("Running in an interactive environment. Scheduling main as a task.")
            loop = asyncio.get_running_loop()
            loop.create_task(main())
        else:
            # Re-raise other RuntimeErrors.
            raise
    except ValueError as e:
        import sys
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
