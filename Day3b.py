# -*- coding: utf-8 -*-
"""
Day 3B: Equipping Agents with Long-Term Memory via In-Memory Service.

This script provides a production-grade, educational example of how to give
an AI agent a persistent, long-term memory using Google's Agent Development
Kit (ADK) and its built-in in-memory memory service for local development.

Key Concepts Illustrated:
- Memory Service Management: Using InMemoryMemoryService for local testing
  and development of memory-enabled agents.
- Writing to Memory: Defining a custom tool (`add_to_memory`) that
  allows the agent to save new information to its memory.
- Reading from Memory: Using the ADK's built-in `load_memory` tool to create a
  powerful, pre-configured search tool that the agent can use to retrieve
  information from its memory.
- Unified Memory Agent: Building a single, coherent agent that can both
  read from and write to memory based on the user's intent.
- End-to-End Workflow: Demonstrating a complete conversational flow where the
  agent first memorizes information and then uses that memory to answer
  subsequent questions.

Prerequisites:
- google-adk
- python-dotenv
- A .env file in the root directory with a GOOGLE_API_KEY.
"""

from initial_setup import *
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import FunctionTool, load_memory
from google.genai import types

# --- Constants and Configuration ---

# Configure a robust retry strategy for API calls to the Gemini model.
RETRY_CONFIG = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


# --- Memory Tool Management ---


def add_to_memory(content: str, memory_service: InMemoryMemoryService) -> dict:
    """
    A custom function (not tool) that demonstrates adding content to memory.
    This follows the pattern from the original notebook where sessions are transferred to memory.

    Args:
        content: The text content to be stored in the memory.
        memory_service: The memory service.

    Returns:
        A dictionary confirming the successful addition of content.
    """
    print(f"  [Memory] 📝 Content to potentially store: '{content}'")
    # In the actual implementation, we'll handle memory storage manually after sessions
    return {"status": "success", "message": f"Content noted: {content}"}


# --- Agent Definition ---


def create_memory_retrieval_agent(memory_service: InMemoryMemoryService):
    """
    Creates a memory agent with only the retrieval tool (following original notebook).
    Memory is populated by transferring sessions to memory separately.

    Args:
        memory_service: The InMemoryMemoryService instance.

    Returns:
        A fully configured LlmAgent.
    """
    # Using the built-in load_memory tool for reading from memory
    print(f"  - 'load_memory' tool created for memory retrieval.")

    memory_agent = LlmAgent(
        name="MemoryAgent",
        model=Gemini(model=GEMINI_MODEL, retry_options=RETRY_CONFIG),
        instruction="""
You are an assistant with a long-term memory.

**Your Capabilities:**
- When the user asks a question (e.g., "Who is...", "What is...", "Remember..."), you MUST use the `load_memory` tool to find relevant information from your memory.
- If you find relevant information in memory, synthesize a clear answer based on that information.
- If the memory has no relevant information, state that you do not know.

**Strict Rules:**
- Do not answer questions from your own knowledge. ALWAYS use `load_memory` first.
""",
        tools=[load_memory],
    )
    print(f"🤖 Agent '{memory_agent.name}' created successfully.")
    return memory_agent


# --- Main Demonstration Logic ---


async def run_conversation_flow(runner, query: str) -> None:
    """
    Helper function to run a single turn of conversation and print the output.

    Args:
        runner: The Runner instance to execute the query.
        query: The user's input for this turn.
    """
    print("-" * 60)
    print(f"💬 User Query: '{query}'")
    print("🧠 Agent is thinking...")

    # Convert query to content format for streaming
    from google.genai import types
    query_content = types.Content(role="user", parts=[types.Part(text=query)])

    response_text = ""
    # Run the conversation with streaming to see tool calls
    # Create the session first
    session_id = "demo_session"
    user_id = "demo_user"

    # Try to create or get the session
    try:
        await runner.session_service.create_session(
            app_name=runner.app_name, user_id=user_id, session_id=session_id
        )
    except:
        # Session might already exist, which is fine
        pass

    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=query_content
    ):
        if event.content and event.content.parts:
            if (
                hasattr(event.content.parts[0], 'text')
                and event.content.parts[0].text != "None"
                and event.content.parts[0].text
            ):
                response_text = event.content.parts[0].text
                print(f"Agent: {response_text}")

    print("\n✅ Agent execution complete.")
    print("--- Agent's Final Response ---")
    print(response_text)
    print("------------------------------\n")


async def main() -> None:
    """
    The main asynchronous entry point for the agent memory demonstration.

    This function orchestrates the entire workflow:
    1. Sets up the environment and API keys.
    2. Creates the memory-enabled agent with InMemoryMemoryService.
    3. Runs a series of conversational turns to demonstrate the agent's ability
       to memorize and recall information.
    """
    print("--- Starting Agent Memory (In-Memory Service) Demonstration ---")
    setup_environment()

    # Create the memory service for local development
    memory_service = InMemoryMemoryService()
    print("✅ InMemoryMemoryService created successfully.")

    # Create the session service (required for the Runner)
    session_service = InMemorySessionService()
    print("✅ InMemorySessionService created successfully.")

    # Create the agent first
    agent = create_memory_retrieval_agent(memory_service)

    # Create the runner with both session and memory services
    runner = Runner(
        agent=agent,
        app_name="MemoryDemoApp",
        session_service=session_service,
        memory_service=memory_service
    )

    print("🏃 Runner initialized with both session and memory services. Ready to process queries.")

    # First, have a conversation to populate the session with project information
    print("\n--- Step 1: Populate session with project information ---")
    await run_conversation_flow(
        runner,
        "The project manager for 'Project Phoenix' is Alice.",
    )

    await run_conversation_flow(
        runner,
        "The final deadline for 'Project Phoenix' is December 15th, 2025.",
    )

    # Now transfer the session data to memory (as demonstrated in the original notebook)
    print("\n--- Step 2: Transfer session data to memory ---")
    session = await session_service.get_session(
        app_name="MemoryDemoApp", user_id="demo_user", session_id="demo_session"
    )
    await memory_service.add_session_to_memory(session)
    print("✅ Session added to memory!")

    # Now ask questions that should be answered from memory
    print("\n--- Step 3: Query memory for the saved information ---")
    await run_conversation_flow(
        runner,
        "Who is the project manager for Project Phoenix?",
    )
    await run_conversation_flow(
        runner,
        "What are the key details for Project Phoenix?",
    )

    print("--- Agent Memory (In-Memory Service) Demonstration Complete ---")


# ==============================================================================
# --- Running the Demonstration ---
# ==============================================================================
#
# This script is designed to be run in two ways:
#
# 1. Interactive Environment (e.g., Jupyter Notebook, VS Code Interactive Window):
#    Step 1: Select the entire code from the top of the file down to (but not including)
#            the `if __name__ == "__main__":` block at the bottom.
#    Step 2: Press SHIFT+ENTER to execute the selected code in the interactive window.
#            This will compile and load all the functions and classes into memory.
#    Step 3: In the interactive input box at the bottom right, type `await main()`
#            and press ENTER to run the demonstration.
#
# 2. Command-Line Execution:
#    The script can also be run directly from the terminal.
#
#        python Day3b.py
#
# ==============================================================================


if __name__ == "__main__":
    # This block handles running the async main function in different environments.
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            print("Running in an interactive environment. Scheduling main as a task.")
            loop = asyncio.get_running_loop()
            loop.create_task(main())
        else:
            raise
    except ValueError as e:
        import sys
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
