# -*- coding: utf-8 -*-
"""
Day 5B: Deploying Your ADK Agent to Production with Vertex AI Agent Engine.

This script provides a production-grade, educational walkthrough of the final
step in the agent development lifecycle: deploying a containerized ADK agent to
Google Cloud's Vertex AI Agent Engine. This process leverages the ADK CLI for
simplified deployment and the Vertex AI SDK for interaction and management.

Key Concepts Illustrated:
- ADK CLI Deployment: Using the `adk deploy agent_engine` command to package,
  containerize, and deploy an agent to Vertex AI Agent Engine.
- Agent Definition for Deployment: Structuring an agent's code and configuration
  within a dedicated directory (`sample_agent`) for ADK CLI processing.
- Vertex AI SDK for Agent Management: Programmatically retrieving, testing, and
  deleting deployed agents using `vertexai.agent_engines`.
- Environment Configuration: Setting up necessary environment variables for
  Google Cloud project and location.
- Cost Management: Emphasizing cleanup of deployed resources to avoid unexpected charges.

Prerequisites:
- A configured Python environment with all packages from `prerequisites.py`.
- Google Cloud SDK (`gcloud`) installed and authenticated.
- ADK CLI installed (usually part of `google-adk[a2a]`).
- A Google Cloud project with the Vertex AI API enabled.
- A `.env` file in the root directory with a valid `GOOGLE_API_KEY` and `GOOGLE_CLOUD_PROJECT`.
"""

import os
import subprocess
import sys
import textwrap
import random
import time
import json

# Ensure initial_setup is run to load environment variables
from initial_setup import setup_environment

# --- Configuration ---

# It's a best practice to load configuration from a centralized place.
# Here, we rely on the setup_environment() function to load from .env.
setup_environment()

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
# Randomly select a location as in the original notebook
regions_list = ["europe-west1", "europe-west4", "us-east4", "us-west1"]
LOCATION = random.choice(regions_list)
AGENT_DIR = "sample_agent"
AGENT_NAME_PREFIX = "adk-weather-agent" # Using a prefix to allow multiple deployments
AGENT_NAME = f"{AGENT_NAME_PREFIX}-{int(time.time())}" # Unique name for each deployment

# --- Helper Functions ---


def print_section_divider(title: str) -> None:
    """Prints a clear visual divider to the console."""
    print("\n" + "=" * 80)
    print(f"| {title.center(76)} |")
    print("=" * 80 + "\n")


def run_shell_command(command: str, description: str, check: bool = True) -> str:
    """
    Executes a shell command and handles errors.

    Args:
        command: The shell command to execute.
        description: A human-readable description of the command's purpose.
        check: If True, raise an exception on non-zero exit code.

    Returns:
        str: The standard output of the command.
    """
    print(f"▶️  Executing: {description}...")
    print(f"   Command: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if result.stdout:
            print(f"   ✅ Success:\n{textwrap.indent(result.stdout, '      ')}")
        if result.stderr:
            print(f"   ⚠️  Stderr:\n{textwrap.indent(result.stderr, '      ')}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error executing command: {description}", file=sys.stderr)
        print(f"   Return Code: {e.returncode}", file=sys.stderr)
        print(f"   Stdout: {e.stdout}", file=sys.stderr)
        print(f"   Stderr: {e.stderr}", file=sys.stderr)
        raise # Re-raise the exception to stop execution
    except Exception as e:
        print(f"   ❌ An unexpected error occurred: {e}", file=sys.stderr)
        raise


def create_agent_deployment_files() -> None:
    """
    Creates the necessary files and directory structure for the agent deployment.
    This mirrors Section 2 of the original notebook.
    """
    print_section_divider("Step 1: Creating Agent Deployment Files")

    # Create agent directory
    run_shell_command(f"mkdir -p {AGENT_DIR}", f"Create agent directory: {AGENT_DIR}")

    # --- requirements.txt ---
    requirements_content = textwrap.dedent("""
    google-adk
    opentelemetry-instrumentation-google-genai
    """)
    with open(os.path.join(AGENT_DIR, "requirements.txt"), "w") as f:
        f.write(requirements_content)
    print(f"   📄 Created {AGENT_DIR}/requirements.txt")

    # --- .env ---
    env_content = textwrap.dedent("""
    # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#global-endpoint
    GOOGLE_CLOUD_LOCATION="global"

    # Set to 1 to use Vertex AI, or 0 to use Google AI Studio
    GOOGLE_GENAI_USE_VERTEXAI=1
    """)
    with open(os.path.join(AGENT_DIR, ".env"), "w") as f:
        f.write(env_content)
    print(f"   📄 Created {AGENT_DIR}/.env")

    # --- agent.py ---
    agent_py_content = '''from google.adk.agents import Agent
import vertexai
import os

vertexai.init(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ["GOOGLE_CLOUD_LOCATION"],
)

def get_weather(city: str) -> dict:
    """
    Returns weather information for a given city.

    This is a TOOL that the agent can call when users ask about weather.
    In production, this would call a real weather API (e.g., OpenWeatherMap).
    For this demo, we use mock data.

    Args:
        city: Name of the city (e.g., "Tokyo", "New York")

    Returns:
        dict: Dictionary with status and weather report or error message
    """
    # Mock weather database with structured responses
    weather_data = {
        "san francisco": {"status": "success", "report": "The weather in San Francisco is sunny with a temperature of 72°F (22°C)."},
        "new york": {"status": "success", "report": "The weather in New York is cloudy with a temperature of 65°F (18°C)."},
        "london": {"status": "success", "report": "The weather in London is rainy with a temperature of 58°F (14°C)."},
        "tokyo": {"status": "success", "report": "The weather in Tokyo is clear with a temperature of 70°F (21°C)."},
        "paris": {"status": "success", "report": "The weather in Paris is partly cloudy with a temperature of 68°F (20°C)."}
    }

    city_lower = city.lower()
    if city_lower in weather_data:
        return weather_data[city_lower]
    else:
        available_cities = ", ".join([c.title() for c in weather_data.keys()])
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available. Try: {available_cities}"
        }

root_agent = Agent(
    name="weather_assistant",
    model="gemini-2.5-flash-lite",  # Fast, cost-effective Gemini model
    description="A helpful weather assistant that provides weather information for cities.",
    instruction="""
    You are a friendly weather assistant. When users ask about the weather:

    1. Identify the city name from their question
    2. Use the get_weather tool to fetch current weather information
    3. Respond in a friendly, conversational tone
    4. If the city isn't available, suggest one of the available cities

    Be helpful and concise in your responses.
    """,
    tools=[get_weather]
)'''
    with open(os.path.join(AGENT_DIR, "agent.py"), "w") as f:
        f.write(agent_py_content)
    print(f"   📄 Created {AGENT_DIR}/agent.py")

    # --- .agent_engine_config.json ---
    agent_engine_config_content = json.dumps({
        "min_instances": 0,
        "max_instances": 1,
        "resource_limits": {"cpu": "1", "memory": "1Gi"}
    }, indent=4)
    with open(os.path.join(AGENT_DIR, ".agent_engine_config.json"), "w") as f:
        f.write(agent_engine_config_content)
    print(f"   📄 Created {AGENT_DIR}/.agent_engine_config.json")


def deploy_agent_with_adk_cli() -> None:
    """
    Deploys the agent to Vertex AI Agent Engine using the ADK CLI.
    This mirrors Section 3.3 of the original notebook.
    """
    print_section_divider("Step 2: Deploying Agent with ADK CLI")

    # The ADK CLI command handles packaging, containerization, and deployment.
    command = (
        f"adk deploy agent_engine --project={PROJECT_ID} --region={LOCATION} "
        f"{AGENT_DIR} --agent_engine_config_file={AGENT_DIR}/.agent_engine_config.json"
    )
    output = run_shell_command(command, "Deploy agent to Vertex AI Agent Engine")

    print("   ✅ Agent deployment initiated. The agent will be available shortly.")
    # Return nothing, since we'll retrieve it by listing all agents

def test_deployed_agent(resource_name: str) -> None:
    """
    Retrieves and tests the deployed agent using the Vertex AI SDK.
    This mirrors Section 4 of the original notebook.
    """
    print_section_divider("Step 3: Testing Deployed Agent")

    try:
        import vertexai
        from vertexai import agent_engines

        print("   SDK imported successfully. Initializing Vertex AI...")
        vertexai.init(project=PROJECT_ID, location=LOCATION)

        print(f"   Retrieving deployed agent: {resource_name}")
        remote_agent = agent_engines.get(resource_name=resource_name)
        print(f"   ✅ Connected to deployed agent: {remote_agent.resource_name}")

        print("\n   Sending query to deployed agent: 'What is the weather in Tokyo?'")
        print("   --- Agent Response ---")
        # The notebook uses async_stream_query, so we'll match that.
        async def query_agent():
            async for item in remote_agent.async_stream_query(
                message="What is the weather in Tokyo?",
                user_id="test_user_day5b",
            ):
                print(f"      {item}")

        # Run the async function
        import asyncio
        asyncio.run(query_agent())
        print("   --- End Agent Response ---")

    except ImportError:
        print("   ❌ Error: `vertexai` SDK is not installed.", file=sys.stderr)
        print("   Please ensure `google-cloud-aiplatform` is installed (it provides `vertexai`).", file=sys.stderr)
        raise
    except Exception as e:
        print(f"   ❌ An error occurred during agent testing: {e}", file=sys.stderr)
        raise

def get_deployed_agent() -> str:
    """
    Retrieves the most recently deployed agent using the Vertex AI SDK.
    This follows the approach in the original notebook.
    """
    print("   Retrieving the most recently deployed agent...")
    try:
        import vertexai
        from vertexai import agent_engines

        print("   SDK imported successfully. Initializing Vertex AI...")
        vertexai.init(project=PROJECT_ID, location=LOCATION)

        # Get the most recently deployed agent
        agents_list = list(agent_engines.list())
        if agents_list:
            remote_agent = agents_list[0]  # Get the first (most recent) agent
            print(f"   ✅ Retrieved deployed agent: {remote_agent.resource_name}")
            return remote_agent.resource_name
        else:
            raise ValueError("❌ No agents found. Please ensure the agent was deployed successfully.")

    except ImportError:
        print("   ❌ Error: `vertexai` SDK is not installed.", file=sys.stderr)
        print("   Please ensure `google-cloud-aiplatform` is installed (it provides `vertexai`).", file=sys.stderr)
        raise
    except Exception as e:
        print(f"   ❌ An error occurred during agent retrieval: {e}", file=sys.stderr)
        raise

def cleanup_deployed_agent(resource_name: str) -> None:
    """
    Deletes the deployed agent from Vertex AI Agent Engine.
    This mirrors Section 6 of the original notebook.
    """
    print_section_divider("Step 4: Cleaning Up Deployed Agent")

    try:
        import vertexai
        from vertexai import agent_engines

        print("   SDK imported successfully. Initializing Vertex AI...")
        vertexai.init(project=PROJECT_ID, location=LOCATION)

        print(f"   Deleting agent: {resource_name}")
        agent_engines.delete(resource_name=resource_name, force=True)
        print("   ✅ Agent successfully deleted.")
    except ImportError:
        print("   ❌ Error: `vertexai` SDK is not installed.", file=sys.stderr)
        print("   Please ensure `google-cloud-aiplatform` is installed (it provides `vertexai`).", file=sys.stderr)
        # Don't re-raise, as cleanup is best-effort
    except Exception as e:
        print(f"   ❌ An error occurred during agent cleanup: {e}", file=sys.stderr)
        # Don't re-raise, as cleanup is best-effort

def main() -> None:
    """
    The main entry point for the deployment script.
    """
    if not PROJECT_ID or PROJECT_ID == "your-gcp-project-id":
        print("❌ Error: GOOGLE_CLOUD_PROJECT environment variable not set or is default.", file=sys.stderr)
        print("   Please set it to your actual Google Cloud Project ID in your .env file or environment.", file=sys.stderr)
        sys.exit(1)

    print_section_divider("Starting ADK Agent Deployment to Vertex AI")
    print(f"Project: {PROJECT_ID}")
    print(f"Location: {LOCATION}")
    print(f"Agent Directory: {AGENT_DIR}")
    print(f"Agent Name: {AGENT_NAME}")

    resource_name = None
    try:
        create_agent_deployment_files()
        deploy_agent_with_adk_cli()

        # Wait for deployment to complete before retrieving
        print("\n⏳ Waiting for deployment to complete (this may take 2-5 minutes)...")
        print("   Checking agent availability...")

        # Try to get the deployed agent with retries (deployment can take a few minutes)
        max_retries = 20  # Wait for up to ~10 minutes (20 * 30 seconds)
        retry_count = 0
        while retry_count < max_retries:
            try:
                resource_name = get_deployed_agent()
                break  # If successful, break out of the loop
            except ValueError as e:
                if "No agents found" in str(e):
                    retry_count += 1
                    print(f"   Agent not ready yet, retrying {retry_count}/{max_retries}...")
                    time.sleep(30)  # Wait 30 seconds before retrying
                else:
                    # If it's a different error, re-raise it
                    raise e

        if resource_name is None:
            raise Exception(f"Agent deployment did not complete after {max_retries * 30} seconds.")

        test_deployed_agent(resource_name)
    except Exception as e:
        print(f"\n🔴 Deployment process failed: {e}", file=sys.stderr)
    finally:
        if resource_name:
            print("\nInitiating cleanup...")
            cleanup_deployed_agent(resource_name)
        else:
            print("\nNo agent was successfully deployed, skipping cleanup.")

    print_section_divider("Deployment Script Finished")
    print("\nTo verify cleanup or check agent status, visit the Vertex AI Agent Builder in the Google Cloud Console.")


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
#        python Day5b.py
#
# ==============================================================================

if __name__ == "__main__":
    # This script orchestrates deployment steps, some of which are synchronous
    # shell commands, and some involve async SDK calls. The main function
    # itself is synchronous, but it calls async functions using asyncio.run().
    # This block handles running the async main function in different environments.
    # `asyncio.run()` is the standard for scripts, but it fails in environments
    # with a running event loop (like Jupyter).
    try:
        main()
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