# -*- coding: utf-8 -*-
"""
Day 4A: Agent Observability with Logs, Traces, and Metrics.

This script provides a production-grade, educational example of how to implement
observability in an AI agent using Google's Agent Development Kit (ADK).
Observability is crucial for debugging, monitoring, and understanding agent
behavior in production environments.

Key Concepts Illustrated:
- Standard Logging: Integrating Python's built-in `logging` module to
  capture application-level events and messages.
- ADK Event Logging: Understanding how the ADK automatically logs detailed
  information about the agent's internal operations.
- Distributed Tracing: Enabling and viewing distributed traces in Google Cloud's
  Vertex AI to visualize the entire lifecycle of an agent request, including
  LLM calls, tool executions, and other internal steps.
- Metrics Collection: Understanding that the ADK automatically collects and
  exports performance metrics (like latency and token counts) to Vertex AI.
- AdkApp Configuration: Using the `AdkApp` class to configure observability
  features like tracing for a production-style agent setup.

Prerequisites:
- google-adk
- python-dotenv
- A .env file in the root directory with a GOOGLE_API_KEY.
- A Google Cloud project with the Vertex AI API enabled.
- Authentication with Google Cloud (e.g., via `gcloud auth application-default login`).
"""

import logging
from initial_setup import *
from google.adk.apps import App
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.tools import google_search
from google.genai import types

# --- Constants and Configuration ---

# Configure a robust retry strategy for API calls to the Gemini model.
RETRY_CONFIG = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# --- Logging Setup ---


def setup_logging() -> None:
    """
    Configures the root logger for the application.

    This function sets up a basic logging configuration that prints messages
    of level INFO and higher to the console. In a production environment, this
    could be configured to write to files, send logs to a centralized logging

    service (like Google Cloud Logging), and use more structured formats (e.g., JSON).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("📝 Standard Python logging configured.")


# --- Agent Definition ---


def create_observable_agent() -> LlmAgent:
    """
    Defines and constructs an agent designed for observability.

    This agent is a standard LlmAgent, but it will be run within an `AdkApp`
    configured for tracing, allowing us to monitor its behavior in detail.

    Returns:
        A fully configured LlmAgent instance.
    """
    observable_agent = LlmAgent(
        name="ObservableAgent",
        model=Gemini(model=GEMINI_MODEL, retry_options=RETRY_CONFIG),
        instruction="""You are a helpful research assistant.
        Use the `google_search` tool to answer questions that require
        up-to-date information. Provide clear and concise answers.""",
        tools=[google_search],
    )
    logging.info(f"🤖 Agent '{observable_agent.name}' created successfully.")
    return observable_agent


# --- Main Demonstration Logic ---


async def main() -> None:
    """
    The main asynchronous entry point for the agent observability demonstration.

    This function orchestrates the entire workflow:
    1. Sets up the environment and logging.
    2. Creates an `AdkApp` with tracing enabled.
    3. Initializes a `Runner` to execute the agent.
    4. Runs a sample query and provides instructions on where to view the
       resulting logs, traces, and metrics.
    """
    print("-- Starting Agent Observability Demonstration --")
    setup_logging()

    # Set up environment variables (this function just configures environment)
    setup_environment()

    # Get project ID from environment variables
    gcp_project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
    gcp_location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")  # Default location

    if not gcp_project_id:
        logging.error("❌ GOOGLE_CLOUD_PROJECT must be set in the environment.")
        print("❌ Environment setup failed: GOOGLE_CLOUD_PROJECT must be set in the environment.")
        print("Please ensure your .env file or environment variables are correctly set.")
        return

    logging.info(f"☁️ Using GCP Project: {gcp_project_id}, Location: {gcp_location}")

    # Create the observable agent
    agent = create_observable_agent()

    # Need to create session service for the runner
    from google.adk.sessions import InMemorySessionService
    session_service = InMemorySessionService()

    # The Runner orchestrates the execution of the agent.
    # For observability, we'll use the runner with proper configuration
    runner = Runner(agent=agent, app_name="ObservabilityDemoApp", session_service=session_service)
    logging.info("🏃 Runner initialized.")

    user_query = "What were the key findings from the latest IPCC report?"
    print("\n" + "-" * 60)
    print(f"💬 User Query: '{user_query}'")
    print("🧠 Agent is thinking...")
    logging.info(f"Executing agent for query: '{user_query}'")

    # Create the session first before running
    await runner.session_service.create_session(
        app_name="ObservabilityDemoApp",
        user_id="demo-user-observability",
        session_id="demo-session-observability"
    )

    # We use `run_async` to execute the agent. The ADK handles the magic of
    # capturing and exporting observability data in the background.
    response_text = ""
    async for event in runner.run_async(
        user_id="demo-user-observability",
        session_id="demo-session-observability",
        new_message=types.Content(role="user", parts=[types.Part(text=user_query)]),
    ):
        if event.content and event.content.parts:
            part_text = getattr(event.content.parts[0], "text", "")
            if part_text and part_text != "None":
                response_text += part_text
                print(f"Agent > {part_text}")

    logging.info("Agent execution complete.")
    print("\n" + "-" * 60)
    print("✅ Agent execution complete.")
    print("\n--- Agent's Final Response ---")
    print(response_text)
    print("------------------------------\n")

    # --- Viewing Observability Data ---
    print("-- 🔎 Where to Find Your Observability Data --")
    print(
        "The ADK has automatically captured logs, traces, and metrics for this run.\n"
        "You can view them in your Google Cloud project:\n"
    )
    print("1. Logs:")
    print("   - The ADK logs detailed events (LLM requests, tool calls, etc.).")
    print(
        f"   - Go to Google Cloud Logging (Logs Explorer) in project '{gcp_project_id}'."
    )
    print(
        "   - Use a query like: `resource.type=\"global\" AND logName=\"projects/.../logs/adk-run-events\"`"
    )
    print("\n2. Traces:")
    print("   - Traces provide a visual timeline of your agent's execution.")
    print(
        f"   - Go to Cloud Trace -> Trace explorer in project '{gcp_project_id}'."
    )
    print(
        "   - You should see a trace for the `ObservabilityDemoApp` run, showing the full journey of the request."
    )
    print("\n3. Metrics:")
    print("   - Metrics give you aggregated data on performance (e.g., latency, token counts).")
    print(
        f"   - Go to Monitoring -> Metrics explorer in project '{gcp_project_id}'."
    )
    print(
        "   - Search for metrics prefixed with `adk/` (e.g., `adk/agent/llm_call/latency`)."
    )
    print("-" * 60)


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
#        python Day4a.py
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
