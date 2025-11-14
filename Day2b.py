# -*- coding: utf-8 -*-
"""
Educational Walkthrough: Advanced Agent Tool Patterns with Google's ADK.

This script serves as a comprehensive, production-grade example for Day 2B of
the Advanced AI Agents course. It demonstrates sophisticated tool usage patterns
that are critical for building robust, real-world agents.

Core Learning Objectives:
1.  **Model Context Protocol (MCP):** Learn how to integrate an agent with
    external services (e.g., a Node.js server) that adhere to the MCP
    standard, enabling agents to leverage tools from other ecosystems.
2.  **Long-Running Operations (Human-in-the-Loop):** Master the technique of
    pausing an agent's execution to wait for human input or other external
    triggers. This is achieved using `tool_context.request_confirmation()`.
3.  **Resumable Workflows:** Understand how to build stateful, resumable
    conversations using the `App`, `Runner`, and `SessionService` components,
    allowing an agent's workflow to be paused and later resumed.

Prerequisites:
- A configured Python environment with all packages from `prerequisites.py`.
- Node.js and the `npx` command available in the system's PATH.
- A `.env` file in the project's root directory containing a valid
  `GOOGLE_API_KEY`.
"""

import uuid
from initial_setup import *
from typing import Any, Dict, List, Optional
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from mcp import StdioServerParameters

# --- Constants and Configuration ---

# The number of shipping containers that, if exceeded, triggers a manual
# approval workflow. This simulates a business rule requiring oversight for
# large-scale logistics operations.
LARGE_ORDER_THRESHOLD = 5

# Defines a robust retry strategy for API calls to the Gemini model.
# This is crucial for production stability, handling transient network issues
# or temporary API rate limits.
# - attempts: Maximum number of retries.
# - exp_base: Multiplier for exponential backoff between retries.
# - initial_delay: The delay before the first retry.
# - http_status_codes: A list of HTTP status codes that trigger a retry.
RETRY_CONFIG = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# A global flag to track the state of the MCP server process. This is used
# to ensure that the external Node.js process is properly terminated during
# cleanup, preventing orphaned processes.
_mcp_server_is_active = False


# --- Utility Functions ---


def print_section_divider(title: Optional[str] = None) -> None:
    """Prints a clear visual divider to the console to separate demo sections."""
    divider = "=" * 80
    if title:
        print(f"\n{divider}\n{title.center(80)}\n{divider}\n")
    else:
        print(f"\n{divider}\n")


def find_approval_request(agent_events: List[Any]) -> Optional[Dict[str, str]]:
    """
    Parses agent events to find a confirmation request from the ADK framework.

    When a tool calls `tool_context.request_confirmation`, the ADK framework
    pauses execution and emits an event containing a special function call named
    `adk_request_confirmation`. This utility scans the event stream for that
    specific function call, which signals that a human-in-the-loop decision
    is required.

    Args:
        agent_events: A list of event objects returned by the agent runner.

    Returns:
        A dictionary containing the `approval_id` and `invocation_id` needed
        to resume the workflow if a request is found; otherwise, returns None.
    """
    for event in agent_events:
        if not (event.content and event.content.parts):
            continue
        for part in event.content.parts:
            if (
                hasattr(part, "function_call")
                and part.function_call
                and part.function_call.name == "adk_request_confirmation"
            ):
                # This is the signal that the agent is paused and waiting for a
                # human response. We capture the necessary IDs to respond.
                return {
                    "approval_id": part.function_call.id,
                    "invocation_id": event.invocation_id,
                }
    return None


def construct_approval_response(
    approval_info: Dict[str, str], approved: bool
) -> types.Content:
    """
    Creates the response message to send back to the agent after a human decision.

    This function constructs a `FunctionResponse` that directly corresponds to the
    `adk_request_confirmation` call. By sending this response back to the agent
    within the same session and invocation, we provide the outcome of the human's
    decision, allowing the paused tool call to resume its execution.

    Args:
        approval_info: The dictionary returned by `find_approval_request`.
        approved: A boolean indicating the human's decision (True for approve).

    Returns:
        A `types.Content` object formatted as a function response that the ADK
        runner can process to resume the workflow.
    """
    confirmation_response = types.FunctionResponse(
        id=approval_info["approval_id"],
        name="adk_request_confirmation",
        response={"confirmed": approved},
    )
    return types.Content(
        role="user", parts=[types.Part(function_response=confirmation_response)]
    )


def print_final_agent_response(agent_events: List[Any]) -> None:
    """
    Finds and prints the agent's final text response from a list of events.

    In a complex workflow, an agent may generate many intermediate events (like
    tool calls). This utility filters through them to find the final, user-facing
    textual response.

    Args:
        agent_events: A list of event objects returned by the agent runner.
    """
    for event in agent_events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    print(f"Agent > {part.text}")


# --- Section 1: Model Context Protocol (MCP) ---

def initialize_mcp_toolset() -> Optional[McpToolset]:
    """
    Creates an MCP toolset by launching and connecting to an external server.

    This function demonstrates a key ADK capability: integrating with external
    tool providers that follow the Model Context Protocol (MCP). It uses `npx`
    to dynamically download and run a pre-built Node.js server that exposes
    a `getTinyImage` tool. The ADK manages the lifecycle of this external
    process.

    Returns:
        An `McpToolset` instance ready to be used by an agent if the connection
        is successful; otherwise, returns None.
    """
    global _mcp_server_is_active
    try:
        # This configuration instructs the ADK on how to start and communicate
        # with the external tool server over its standard I/O streams.
        mcp_toolset = McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="npx",
                    args=["-y", "@modelcontextprotocol/server-everything"],
                    tool_filter=["getTinyImage"],  # Only expose this specific tool
                ),
                timeout=15,  # Timeout for establishing the connection
            )
        )
        _mcp_server_is_active = True
        print(
            "  - MCP Toolset created for Everything Server, exposing 'getTinyImage'."
        )
        return mcp_toolset
    except Exception as e:
        print(
            f"  - WARNING: Failed to create MCP toolset. This demo will be skipped. Error: {e}"
        )
        _mcp_server_is_active = False
        return None


def build_mcp_image_agent(mcp_toolset: Optional[McpToolset]) -> LlmAgent:
    """
    Builds an LlmAgent equipped with the MCP toolset for image generation.

    Args:
        mcp_toolset: The MCP toolset that provides the `getTinyImage` tool.

    Returns:
        An `LlmAgent` instance configured with the specified tools.
    """
    image_agent = LlmAgent(
        model=Gemini(model=GEMINI_MODEL, retry_options=RETRY_CONFIG),
        name="image_agent",
        instruction="""You are an image generation assistant. When users ask for an image,
        use the `getTinyImage` tool. Describe what the image shows in simple terms.
        Explain that you can only generate new 16x16 pixel images.""",
        tools=[mcp_toolset] if mcp_toolset else [],
    )
    print(f"  - Image agent '{image_agent.name}' built.")
    return image_agent


async def demonstrate_mcp_integration() -> None:
    """
    Orchestrates the demonstration of agent integration with an external MCP server.
    """
    print("--- (1/3) Demonstrating MCP Integration ---")
    mcp_toolset = initialize_mcp_toolset()

    if not mcp_toolset:
        print("  - Skipping MCP demo as the toolset could not be created.")
        return

    try:
        agent = build_mcp_image_agent(mcp_toolset)
        app = App(name="mcp_app", root_agent=agent)
        session_service = InMemorySessionService()
        runner = Runner(app=app, session_service=session_service)

        query = "Generate a tiny test image."
        print(f"\n🖼️  Query: '{query}'")
        print("🧠 Agent is thinking...")

        # Create the session before running the agent.
        session_id = "mcp_session"
        await session_service.create_session(
            app_name=app.name, user_id="test_user", session_id=session_id
        )

        events = []
        async for event in runner.run_async(
            user_id="test_user",
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=query)]),
        ):
            events.append(event)

        print("\n✅ MCP Agent execution complete.")
        print_final_agent_response(events)

    except Exception as e:
        print(f"❌ Error in MCP integration demo: {e}")
    finally:
        # Explicitly close the toolset to ensure its subprocess is terminated.
        if mcp_toolset:
            print("  - Closing MCP toolset to clean up resources...")
            await mcp_toolset.close()

        # Reset our global tracking flag.
        global _mcp_server_is_active
        if _mcp_server_is_active:
            _mcp_server_is_active = False
            print("  - MCP server resources cleaned up.")


# --- Section 2: Long-Running Operations (Human-in-the-Loop) ---


def place_shipping_order(num_containers: int, destination: str, tool_context: ToolContext,) -> Dict[str, Any]:
    """
    A tool that places a shipping order, pausing for human approval on large orders.
    This function demonstrates a robust pattern for human-in-the-loop workflows.

    1.  **First Call:** The agent invokes the tool. If the order requires
        approval, the tool calls `tool_context.request_confirmation()`
        to pause execution.

    2.  **Second Call (Resumption):** After the human responds, the ADK re-invokes
        this tool. The tool now proceeds based on the human's decision, which is
        available in `tool_context.tool_confirmation.confirmed`.

    Args:
        num_containers: The number of containers to ship.
        destination: The destination port or city.
        tool_context: The ADK-provided context, giving access to the
        confirmation result.

    Returns:
        A dictionary containing the order status and a unique order ID.
    """

    # Stage 1: Initial tool call from the agent.
    if not tool_context.tool_confirmation:
        if num_containers > LARGE_ORDER_THRESHOLD:
            # This is a large order, so we request human approval.
            print(f"   [Tool Internals] Pausing: Large order of {num_containers} containers needs approval.")
            
            tool_context.request_confirmation(
                hint=f"Large order: {num_containers} containers to {destination}. Requires approval."
            )
            
            # NOTE: We generate an order ID here for the initial response. In a
            # real system, this ID would be persisted in a database.
            return {"status": "pending_approval", "order_id": f"ORD-{uuid.uuid4().hex[:6].upper()}"}
        else:
            # This is a small order, so we can auto-approve it.
            order_id = f"ORD-{uuid.uuid4().hex[:6].upper()}"
            print(f"   [Tool Internals] Auto-approving small order {order_id}.")
            return {"status": "approved_auto", "order_id": order_id}
    
    # Stage 2: Tool is re-invoked after human confirmation.
    else:
        # NOTE: In a real production system, you would retrieve the original
        # order ID from a persistent database using session information.
        # For this educational example, we generate a new ID upon completion
        # to keep the focus on the pause/resume mechanism itself.
        order_id = f"ORD-{uuid.uuid4().hex[:6].upper()}"
        
        if tool_context.tool_confirmation.confirmed:
            print(f"   [Tool Internals] Resuming: Order {order_id} was approved by human.")
            return {"status": "approved_manual", "order_id": order_id}
        else:
            print(f"   [Tool Internals] Resuming: Order {order_id} was rejected by human.")
            return {"status": "rejected", "order_id": order_id}

def build_resumable_shipping_app() -> App:
    """
    Creates a resumable App for the shipping agent.

    For a workflow to be resumable (i.e., to handle pauses for human input),
    it must be wrapped in an `App` with `is_resumable=True`. This tells the
    ADK to persist session state, allowing the `Runner` to pause and later
    resume the conversation.

    Returns:
        A resumable `App` instance containing the shipping agent.
    """
    shipping_agent = LlmAgent(
        name="shipping_agent",
        model=Gemini(model=GEMINI_MODEL, retry_options=RETRY_CONFIG),
        instruction="""You are a shipping coordinator. Use `place_shipping_order`.
        If the status is 'pending_approval', inform the user that the order requires
        manual approval and is now pending. Otherwise, report the final status
        (e.g., 'approved_auto', 'approved_manual', 'rejected') and the order ID.""",
        tools=[FunctionTool(func=place_shipping_order)],
    )
    return App(
        name="shipping_coordinator",
        root_agent=shipping_agent,
        resumability_config=ResumabilityConfig(is_resumable=True),
    )


async def run_shipping_workflow(query: str, auto_approve: bool) -> None:
    """
    Orchestrates a full shipping workflow, simulating a human decision.

    This function demonstrates the complete end-to-end flow of a resumable
    operation, from the initial user query to the final agent response after
    a potential human intervention.

    Args:
        query: The user's initial shipping request (e.g., "Ship 10 containers").
        auto_approve: A boolean to simulate the human's approval decision if
                      the workflow pauses.
    """
    print_section_divider(f"Workflow for query: '{query}'")
    print(f"User > {query}")

    # A SessionService is required for resumable apps to store conversation state.
    session_service = InMemorySessionService()
    shipping_app = build_resumable_shipping_app()
    shipping_runner = Runner(app=shipping_app, session_service=session_service)
    session_id = f"order_{uuid.uuid4().hex[:8]}"
    await session_service.create_session(
        app_name=shipping_app.name, user_id="test_user", session_id=session_id
    )

    # --- Step 1: Initial agent call ---
    print("\n--- Agent Execution: Part 1 (Initial Request) ---")
    initial_events = []
    async for event in shipping_runner.run_async(
        user_id="test_user",
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=query)]),
    ):
        initial_events.append(event)

    print_final_agent_response(initial_events)
    approval_info = find_approval_request(initial_events)

    # --- Step 2: Check if approval is needed and resume if necessary ---
    if approval_info:
        print("\n--- Human Intervention Required ---")
        print(f"🤔 Human Decision: {'APPROVE ✅' if auto_approve else 'REJECT ❌'}")
        print("\n--- Agent Execution: Part 2 (Resuming Workflow) ---")

        resume_events = []
        async for event in shipping_runner.run_async(
            user_id="test_user",
            session_id=session_id,
            new_message=construct_approval_response(approval_info, auto_approve),
            # Resuming the *same invocation* is the key to continuing the
            # paused tool call correctly.
            invocation_id=approval_info["invocation_id"],
        ):
            resume_events.append(event)
        print_final_agent_response(resume_events)

    print(f"\n--- Workflow for '{query}' Complete ---")


async def demonstrate_long_running_operations() -> None:
    """
    Runs a series of demonstrations for long-running operations, covering
    different human-in-the-loop approval scenarios.
    """
    print("--- (2/3) Demonstrating Long-Running Operations (Human-in-the-Loop) ---")
    # Demo 1: Small order that is auto-approved without human intervention.
    await run_shipping_workflow("Ship 3 containers to Singapore", auto_approve=True)
    # Demo 2: Large order that requires and receives manual approval.
    await run_shipping_workflow("Ship 10 containers to Rotterdam", auto_approve=True)
    # Demo 3: Large order that requires and is denied manual approval.
    await run_shipping_workflow("Ship 8 containers to Los Angeles", auto_approve=False)


# --- Section 3: Production-Ready Example ---
def demonstrate_production_readiness() -> None:
    """
    Placeholder for the third demonstration, highlighting that the previous
    section already constitutes a production-style pattern.
    """
    print("--- (3/3) Production-Ready Example ---")
    print(
        "The 'Long-Running Operations' demo in the previous section is a complete,\n" 
        "production-style example. It correctly uses a resumable App, a session\n" 
        "service, and the two-stage tool pattern to manage a human-in-the-loop\n" 
        "workflow, which is a common requirement in production agent systems."
    )

# --- Main Execution ---
async def main() -> None:
    """
    The main asynchronous entry point for running all advanced agent demos.
    """
    print_section_divider("Starting Advanced Agent Tool Patterns Demo")
    setup_environment()

    await demonstrate_mcp_integration()
    print_section_divider()
    await demonstrate_long_running_operations()
    print_section_divider()
    demonstrate_production_readiness()

    print_section_divider("Advanced Agent Tool Patterns Demo Complete")

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
#        python Day2b.py
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
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
