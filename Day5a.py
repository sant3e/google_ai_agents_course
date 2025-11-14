# -*- coding: utf-8 -*-
"""
Day 5A: Agent-to-Agent (A2A) Communication.

This script provides a production-grade, educational example of how to build
multi-agent systems where agents communicate with each other using the
Agent-to-Agent (A2A) protocol, facilitated by Google's Agent Development Kit (ADK).

Key Concepts Illustrated:
- A2A Protocol: Understanding how agents can expose themselves as services
  and be consumed by other agents, enabling complex, decoupled systems.
- Remote Agent as a Sub-Agent: Using the `RemoteA2aAgent` to wrap a remote agent,
  making it available as a sub-agent for a local agent.
- Service Discovery: Using agent cards for discovering remote agent capabilities.
- Multi-Agent Collaboration: Building a simple but powerful system where one
  agent (a "manager") delegates a specialized task (math calculation) to
  another agent (a "specialist").
- Asynchronous Agent Server: Although simulated here, this script lays the
  groundwork for understanding how you would run an agent as a persistent
  background service that other agents can call.

Prerequisites:
- google-adk[a2a]
- python-dotenv
- A .env file in the root directory with a GOOGLE_API_KEY.
"""

from initial_setup import *
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio

# --- Constants and Configuration ---

# Configure a robust retry strategy for API calls to the Gemini model.
RETRY_CONFIG = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# In a real-world scenario, this URL would point to a deployed agent server.
REMOTE_AGENT_URL = "http://localhost:8001"
AGENT_CARD_URL = f"{REMOTE_AGENT_URL}/.well-known/agent-card.json"


# --- Agent Definitions ---


def create_math_specialist_agent() -> LlmAgent:
    """
    Defines the specialist agent that performs calculations.

    In a real system, this agent would be running as a separate service.
    Here, we define it to understand its role and instruction set. Its key
    characteristic is that it's designed to be called by another agent, not
    directly by a human.

    Returns:
        A fully configured LlmAgent designed for calculation tasks.
    """
    math_agent = LlmAgent(
        name="math_specialist_agent",
        model=Gemini(model=GEMINI_MODEL, retry_options=RETRY_CONFIG),
        instruction="""You are a math specialist. You will be given a math
        problem. Solve it and return only the final answer without any
        explanation, preamble, or formatting.""",
    )
    print(f"🤖 Specialist Agent '{math_agent.name}' defined.")
    return math_agent


def create_manager_agent_with_remote_agent(remote_agent) -> LlmAgent:
    """
    Defines the manager agent that delegates tasks to specialists.

    This agent's primary role is to understand the user's intent and delegate
    the task to the appropriate specialist agent, in this case, the remote
    `math_specialist_agent`.

    Args:
        remote_agent: A `RemoteA2aAgent` instance that represents the remote
                      math specialist agent.

    Returns:
        A fully configured LlmAgent designed for task delegation.
    """
    # Import inside function to avoid issues when A2A is not available
    from google.adk.agents import LlmAgent
    from google.adk.models.google_llm import Gemini
    from google.genai import types

    RETRY_CONFIG = types.HttpRetryOptions(
        attempts=5,
        exp_base=7,
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504],
    )

    manager_agent = LlmAgent(
        name="manager_agent",
        model=Gemini(model=GEMINI_MODEL, retry_options=RETRY_CONFIG),
        instruction="""You are a manager agent. Your job is to delegate tasks
        to specialist agents. When a user asks a math-related question, you
        MUST use the math_specialist_agent sub-agent to solve it.
        Present the final answer to the user clearly.""",
        sub_agents=[remote_agent],
    )
    print(f"🤖 Manager Agent '{manager_agent.name}' created.")
    return manager_agent


# --- Main Demonstration Logic ---


async def main() -> None:
    """
    The main asynchronous entry point for the A2A communication demonstration.

    This function orchestrates the entire workflow:
    1. Sets up the environment.
    2. Checks for A2A dependencies.
    3. If available, starts the specialist agent server, creates a RemoteA2aAgent client,
       and demonstrates A2A communication.
    4. If not available, runs a simplified demonstration.
    """
    print("--- Starting Agent-to-Agent (A2A) Communication Demonstration ---")
    setup_environment()

    # Check if a2a dependencies are available
    try:
        from google.adk.agents.remote_a2a_agent import RemoteA2aAgent, AGENT_CARD_WELL_KNOWN_PATH
        from google.adk.a2a.utils.agent_to_a2a import to_a2a
        import subprocess
        import time
        import requests
        import sys
        import tempfile
        import os

        print("✅ A2A dependencies are available")

        # 1. Define the specialist agent that we want to expose via A2A.
        math_specialist = create_math_specialist_agent()

        # 2. Create the math specialist agent server file
        print("📝 Creating math specialist agent server...")

        math_agent_code = f'''import os
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.genai import types

# Configure retry options
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Create the math specialist agent
math_specialist_agent = LlmAgent(
    model=Gemini(model="{GEMINI_MODEL}", retry_options=retry_config),
    name="math_specialist_agent",
    instruction=\"\"\"You are a math specialist. You will be given a math
    problem. Solve it and return only the final answer without any
    explanation, preamble, or formatting.\"\"\",
)

if __name__ == "__main__":
    # Create the A2A app and run it
    import uvicorn
    app = to_a2a(math_specialist_agent, port=8001)
    uvicorn.run(app, host="localhost", port=8001)
'''

        # Write the agent code to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='_math_agent_server.py', delete=False) as f:
            f.write(math_agent_code)
            temp_server_path = f.name

        # 3. Start the specialist agent server in background
        print("🚀 Starting Math Specialist Agent server...")
        server_process = subprocess.Popen(
            [
                sys.executable, "-c",
                f"import sys; sys.path.append('.'); exec(open('{temp_server_path}').read())"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ},  # Pass environment variables (including GOOGLE_API_KEY)
        )

        print("   Waiting for server to be ready...")

        # Wait for server to start (poll until it responds)
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                response = requests.get(AGENT_CARD_URL, timeout=2)
                if response.status_code == 200:
                    print(f"\n✅ Math Specialist Agent server is running!")
                    print(f"   Server URL: {REMOTE_AGENT_URL}")
                    print(f"   Agent card: {AGENT_CARD_URL}")
                    break
            except requests.exceptions.RequestException:
                time.sleep(2)
                print(".", end="", flush=True)
        else:
            print("\n❌ Math Specialist Agent server failed to start. Please check the error.")
            # Try to get error output
            stderr_output = server_process.stderr.read() if server_process.poll() is not None else b""
            if stderr_output:
                print(f"Error output: {stderr_output.decode()}")
            return
        print()

        # 4. Create the RemoteA2aAgent. This acts as a client-side proxy to the remote agent.
        try:
            remote_agent = RemoteA2aAgent(
                name="math_specialist_agent",
                description="Remote math specialist agent from external vendor that provides calculation services.",
                # Point to the agent card URL - this is where the A2A protocol metadata lives
                agent_card=AGENT_CARD_URL,
            )
            print(
                f"🔗 RemoteA2aAgent created to connect to '{remote_agent.name}' at {REMOTE_AGENT_URL}."
            )
        except Exception as e:
            print(f"❌ Error creating RemoteA2aAgent: {e}")
            return

        # 5. Define the manager agent and give it the remote agent as a sub-agent.
        manager_agent = create_manager_agent_with_remote_agent(remote_agent=remote_agent)

        # 6. Set up the runner to execute the top-level (manager) agent.
        runner = Runner(
            agent=manager_agent,
            app_name="A2ADemoApp",
            session_service=InMemorySessionService()
        )
        print("🏃 Runner initialized to run the ManagerAgent.")

        user_query = "What is 1024 * (256 - 128)?"
        print("\n" + "-" * 60)
        print(f"💬 User Query: '{user_query}'")
        print("🧠 ManagerAgent is thinking...")

        # We use `run_async` here to get the response from the agent with A2A communication
        response = ""
        try:
            # Create the session before running the agent
            session = await runner.session_service.create_session(
                app_name="A2ADemoApp",
                user_id="demo_user",
                session_id="demo_session"
            )

            async for event in runner.run_async(
                user_id="demo_user",
                session_id="demo_session",
                new_message=types.Content(parts=[types.Part(text=user_query)])
            ):
                if event.is_final_response() and event.content:
                    for part in event.content.parts:
                        if hasattr(part, "text"):
                            response += part.text

            print("\n" + "-" * 60)
            print("✅ A2A execution complete.")
            print("\n--- ManagerAgent's Final Response ---")
            print(response)
            print("------------------------------------")
            print(
                "\n💡 Review the output above to see how the 'ManagerAgent' communicated\n"
                "   with the 'MathSpecialistAgent' via A2A protocol to get the answer."
            )
        except Exception as e:
            print(f"❌ Error during A2A communication: {e}")
        finally:
            # Clean up - terminate the server process
            server_process.terminate()
            server_process.wait()
            # Clean up the temporary file
            os.unlink(temp_server_path)
    except ImportError as e:
        print(f"❌ A2A dependencies not available. Error: {e}")
        print("To install A2A dependencies, run: pip install google-adk[a2a]")
        print("For now, running a simplified version without A2A functionality.")

        # Fallback to simplified version without A2A
        manager_agent = LlmAgent(
            name="manager_agent",
            model=Gemini(model=GEMINI_MODEL, retry_options=RETRY_CONFIG),
            instruction="""You are a manager agent. For math-related questions,
            solve them directly as a demonstration of how A2A communication would work
            if the remote agent was available. Present the final answer to the user clearly.""",
        )
        runner = Runner(
            agent=manager_agent,
            app_name="A2ADemoApp",
            session_service=InMemorySessionService()
        )
        print("🏃 Runner initialized to run the ManagerAgent.")

        user_query = "What is 1024 * (256 - 128)?"
        print("\n" + "-" * 60)
        print(f"💬 User Query: '{user_query}'")
        print("🧠 ManagerAgent is thinking...")

        response = await runner.run_debug(user_query)

        print("\n" + "-" * 60)
        print("✅ A2A execution complete (simplified version).")
        print("\n--- ManagerAgent's Final Response ---")
        print(response)
        print("------------------------------------")
        print(
            "\n💡 In a real A2A implementation with proper dependencies, the 'ManagerAgent'\n"
            "   would communicate with a remote 'MathSpecialistAgent' via the A2A protocol."
        )


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
#        python Day5a.py
#
# ==============================================================================

if __name__ == "__main__":
    # This block handles running the async main function in different environments.
    # `asyncio.run()` is the standard for scripts, but it fails in environments
    # with a running event loop (like Jupyter).
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            # If in a Jupyter notebook or other interactive environment, schedule
            # the main function as a task on the existing loop.
            print(
                "Running in an interactive environment. Scheduling main as a task."
            )
            loop = asyncio.get_running_loop()
            loop.create_task(main())
        else:
            # Re-raise other RuntimeErrors.
            raise
    except ValueError as e:
        import sys
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
