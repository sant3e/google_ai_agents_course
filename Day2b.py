# -*- coding: utf-8 -*-
"""
Day 2B: Advanced Agent Tool Patterns.

This script provides an educational walkthrough of advanced agent tool patterns
using Google's Agent Development Kit (ADK), including Model Context Protocol (MCP),
long-running operations, and resumable workflows.

Key Concepts Illustrated:
- Model Context Protocol (MCP): Connecting to external systems via standardized protocol
- Long-Running Operations: Pausing agent execution for human input or time-consuming tasks
- Resumable Workflows: Maintaining state across conversation breaks for persistent workflows
- ToolContext: Handling approval and confirmation flows in tools

Prerequisites:
- google-adk
- python-dotenv
- mcp
- Node.js (for MCP servers)
- A .env file in the root directory with a GOOGLE_API_KEY.
"""

from typing import Dict, Any, Optional
import uuid
from initial_setup import GEMINI_MODEL, setup_environment, asyncio, sys

from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner, InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool
from mcp import StdioServerParameters

# --- Constants and Configuration ---

LARGE_ORDER_THRESHOLD = 5
"""Threshold for shipping orders that require human approval."""

RETRY_CONFIG = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


# --- Section 1: Model Context Protocol (MCP) ---

# Global variable to track MCP server state
_mcp_server_active = False

async def cleanup_mcp_resources():
    """Clean up MCP server resources."""
    global _mcp_server_active
    if _mcp_server_active:
        try:
            # Give time for cleanup
            await asyncio.sleep(0.2)
            _mcp_server_active = False
        except Exception:
            pass

def create_mcp_image_server() -> McpToolset:
    """
    Creates an MCP toolset for the Everything MCP server.

    This MCP server provides a `getTinyImage` tool that returns a simple test image
    (16x16 pixels, Base64-encoded). This is used for demonstration purposes.
    In production, you would connect to servers for Google Maps, Slack, GitHub, etc.

    Returns:
        McpToolset: The configured MCP toolset for image generation
    """
    global _mcp_server_active
    try:
        mcp_image_server = McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="npx",  # Run MCP server via npx
                    args=[
                        "-y",  # Argument for npx to auto-confirm install
                        "@modelcontextprotocol/server-everything",
                    ],
                    tool_filter=["getTinyImage"],
                ),
                timeout=15,  # Reduced timeout to avoid hanging
            )
        )
        _mcp_server_active = True
        print("  - MCP Tool created for Everything Server with getTinyImage tool")
        return mcp_image_server
    except Exception as e:
        print(f"  - Warning: Failed to create MCP tool: {e}")
        _mcp_server_active = False
        # Return None if MCP server fails to start
        return None


def create_image_agent() -> LlmAgent:
    """
    Creates an image generation agent that uses MCP tools.

    This agent demonstrates how to integrate MCP tools into an agent's workflow.

    Returns:
        LlmAgent: An agent configured to generate tiny images via MCP
    """
    mcp_tool = create_mcp_image_server()
    tools = [mcp_tool] if mcp_tool else []
    
    image_agent = LlmAgent(
        model=Gemini(model=GEMINI_MODEL, retry_options=RETRY_CONFIG),
        name="image_agent",
        instruction="""You are an image generation assistant. When users ask for images:
        1. Use the getTinyImage tool to generate a 16x16 pixel image
        2. Describe what the image shows in simple terms
        3. If they ask for modifications, explain that you can only generate new images

        Note: The tool returns base64-encoded image data.""",
        tools=tools,
    )
    print(f"  - Image agent '{image_agent.name}' created with MCP integration")
    return image_agent


async def demo_mcp_integration() -> None:
    """
    Demonstrates MCP integration with the image generation agent.
    """
    print("--- (1/3) Demonstrating MCP Integration ---")
    try:
        agent = create_image_agent()
        runner = InMemoryRunner(agent=agent)
        print("  - Runner initialized for MCP demo.")
        
        query = "Generate a tiny test image"
        print(f"\n🖼️  Query: '{query}'")
        print("🧠 Agent is thinking...")

        # Use a timeout to prevent hanging
        try:
            response = await asyncio.wait_for(runner.run_debug(query), timeout=30)
            print("\n✅ Agent execution complete.")
            print("--- Image Generation Result ---")
            print(response)
            print("------------------------------")
        except asyncio.TimeoutError:
            print("\n⏱️  MCP demo timed out - this is expected if MCP server is unavailable")
        except Exception as e:
            print(f"\n⚠️  MCP demo completed with warnings: {e}")
    except Exception as e:
        print(f"❌ Error in MCP integration demo: {e}")
        print("⚠️  This may be due to MCP server connection issues.")
    finally:
        # Clean up MCP resources
        await cleanup_mcp_resources()


# --- Section 2: Long-Running Operations (Human-in-the-Loop) ---

def place_shipping_order(num_containers: int, destination: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Place a shipping order with approval logic for long-running operations.

    This function demonstrates how tools can pause execution for human input.
    If the number of containers exceeds the threshold, it requests human approval.
    The function can be called multiple times - once initially and potentially
    again after human approval is provided.

    Args:
        num_containers: Number of containers to ship
        destination: Destination port/city
        tool_context: ADK tool context for requesting confirmation

    Returns:
        dict: Order status and details
    """
    # Check if this is the first call or a resumed call
    if not tool_context.tool_confirmation:
        # First call - check if approval is needed
        if num_containers > LARGE_ORDER_THRESHOLD:
            # Large order - request human approval
            tool_context.request_confirmation(
                hint=f"Large order detected: {num_containers} containers to {destination}. Requires approval.",
                payload={
                    "title": "Shipping Order Approval",
                    "description": f"Approve shipping {num_containers} containers to {destination}"
                },
            )
            return {
                "status": "pending",
                "message": f"Awaiting approval for {num_containers} containers to {destination}",
                "order_id": f"ORD-{num_containers}-PENDING",
            }
        else:
            # Small order - auto-approve
            return {
                "status": "approved",
                "message": f"Auto-approved small order: {num_containers} containers to {destination}",
                "order_id": f"ORD-{num_containers}-AUTO",
            }
    else:
        # Resumed call - check the approval decision
        if tool_context.tool_confirmation.confirmed:
            # Human approved - complete the order
            return {
                "status": "approved",
                "message": f"Human approved: {num_containers} containers to {destination}",
                "order_id": f"ORD-{num_containers}-HUMAN",
            }
        else:
            # Human rejected - cancel the order
            return {
                "status": "rejected",
                "message": f"Human rejected: {num_containers} containers to {destination}",
                "order_id": f"ORD-{num_containers}-CANCELLED",
            }


def create_shipping_agent() -> LlmAgent:
    """
    Creates a shipping agent that uses long-running operations.
    
    This agent demonstrates how to handle workflows that need to pause
    for human input before completing.
    
    Returns:
        LlmAgent: An agent configured to handle shipping orders with approval logic
    """
    shipping_agent = LlmAgent(
        name="shipping_agent",
        model=Gemini(model=GEMINI_MODEL, retry_options=RETRY_CONFIG),
        instruction="""You are a shipping coordinator assistant.

    When users request to ship containers:
    1. Use the place_shipping_order tool with the number of containers and destination
    2. If the order status is 'pending', inform the user that approval is required
    3. After receiving the final result, provide a clear summary including:
        - Order status (approved/rejected)
        - Order ID (if available)
        - Number of containers and destination
    4. Keep responses concise but informative
    """,
        tools=[FunctionTool(func=place_shipping_order)],
    )
    print(f"  - Shipping Agent '{shipping_agent.name}' created with long-running operations capability")
    return shipping_agent


def create_shipping_app() -> App:
    """
    Creates a resumable app for the shipping agent.
    
    This is necessary for long-running operations to maintain state
    between the initial call and the approval/resume call.
    
    Returns:
        App: A resumable app wrapping the shipping agent
    """
    shipping_agent_instance = create_shipping_agent()
    shipping_app = App(
        name="shipping_coordinator",
        root_agent=shipping_agent_instance,
        resumability_config=ResumabilityConfig(is_resumable=True),
    )
    print(f"  - Resumable shipping app '{shipping_app.name}' created with state persistence")
    return shipping_app


def check_for_approval(events) -> Optional[Dict[str, str]]:
    """
    Check if events contain an approval request.
    
    Args:
        events: The events returned by the agent runner
        
    Returns:
        dict with approval details or None if no approval is requested
    """
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if (
                    hasattr(part, 'function_call') and 
                    part.function_call and 
                    part.function_call.name == "adk_request_confirmation"
                ):
                    return {
                        "approval_id": part.function_call.id,
                        "invocation_id": event.invocation_id,
                    }
    return None


def print_agent_response(events) -> None:
    """
    Print the agent's text responses from events.
    
    Args:
        events: The events returned by the agent runner
    """
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    print(f"Agent > {part.text}")


def create_approval_response(approval_info: Dict[str, str], approved: bool):
    """
    Create an approval response message.
    
    Args:
        approval_info: Dictionary containing approval ID
        approved: Whether the action is approved
        
    Returns:
        types.Content: The approval response content
    """
    confirmation_response = types.FunctionResponse(
        id=approval_info["approval_id"],
        name="adk_request_confirmation",
        response={"confirmed": approved},
    )
    return types.Content(
        role="user", 
        parts=[types.Part(function_response=confirmation_response)]
    )


async def run_shipping_workflow(query: str, auto_approve: bool = True) -> None:
    """
    Runs a shipping workflow with approval handling.

    Args:
        query: User's shipping request
        auto_approve: Whether to auto-approve large orders (simulates human decision)
    """
    print(f"\n{'='*60}")
    print(f"User > {query}")

    # Create session service and runner
    session_service = InMemorySessionService()
    shipping_app = create_shipping_app()
    shipping_runner = Runner(
        app=shipping_app,
        session_service=session_service,
    )

    # Generate unique session ID
    session_id = f"order_{uuid.uuid4().hex[:8]}"

    # Create session
    await session_service.create_session(
        app_name="shipping_coordinator", user_id="test_user", session_id=session_id
    )

    query_content = types.Content(role="user", parts=[types.Part(text=query)])
    events = []

    try:
        # Send initial request to the Agent with timeout
        try:
            async for event in shipping_runner.run_async(
                user_id="test_user", session_id=session_id, new_message=query_content
            ):
                events.append(event)
        except asyncio.TimeoutError:
            print("⏱️  Initial request timed out")
            return

        # Check if approval is requested
        approval_info = check_for_approval(events)

        if approval_info:
            print(f"⏸️  Pausing for approval...")
            print(f"🤔 Human Decision: {'APPROVE ✅' if auto_approve else 'REJECT ❌'}\n")

            # Resume the agent with approval decision with timeout
            try:
                async for event in shipping_runner.run_async(
                    user_id="test_user",
                    session_id=session_id,
                    new_message=create_approval_response(approval_info, auto_approve),
                    invocation_id=approval_info["invocation_id"],  # Critical: same invocation_id tells ADK to RESUME
                ):
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                print(f"Agent > {part.text}")
            except asyncio.TimeoutError:
                print("⏱️  Approval response timed out")
        else:
            print_agent_response(events)
    except asyncio.TimeoutError:
        print(f"⏱️  Shipping workflow timed out")
    except Exception as e:
        print(f"❌ Error in shipping workflow: {e}")
        # Continue with other demos even if this one fails
    finally:
        # Clean up resources
        try:
            # Give time for cleanup
            await asyncio.sleep(0.1)
        except Exception:
            pass  # Ignore cleanup errors

    print(f"{'='*60}\n")


async def demo_long_running_operations() -> None:
    """
    Demonstrates long-running operations with approval workflows.
    """
    print("--- (2/3) Demonstrating Long-Running Operations ---")
    
    # Demo 1: Small order - auto-approved
    await run_shipping_workflow("Ship 3 containers to Singapore")

    # Demo 2: Large order - auto-approved for demonstration
    await run_shipping_workflow("Ship 10 containers to Rotterdam", auto_approve=True)

    # Demo 3: Large order - rejected for demonstration
    await run_shipping_workflow("Ship 8 containers to Los Angeles", auto_approve=False)


# --- Section 3: Resumable Workflows ---

def place_shipping_order_with_tracking(num_containers: int, destination: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Enhanced shipping order with tracking and multi-step approval.

    Args:
        num_containers: Number of containers to ship
        destination: Destination port/city
        tool_context: ADK tool context for requesting confirmation

    Returns:
        dict: Order status and details with step information
    """
    # Check if this is the first call or a resumed call
    if not tool_context.tool_confirmation:
        # First call - check if approval is needed
        if num_containers > LARGE_ORDER_THRESHOLD:
            # Large order - request human approval
            tool_context.request_confirmation(
                hint=f"Large order detected: {num_containers} containers to {destination}. Requires approval.",
                payload={
                    "title": "Shipping Order Approval",
                    "description": f"Approve shipping {num_containers} containers to {destination}"
                },
            )
            return {
                "status": "pending_approval",
                "message": f"Awaiting approval for {num_containers} containers to {destination}",
                "order_id": f"ORD-{num_containers}-PENDING",
                "step": "approval_required",
            }
        else:
            # Small order - auto-approve
            return {
                "status": "approved",
                "message": f"Auto-approved small order: {num_containers} containers to {destination}",
                "order_id": f"ORD-{num_containers}-AUTO",
                "step": "completed",
            }
    else:
        # Resumed call - check the approval decision
        if tool_context.tool_confirmation.confirmed:
            # Human approved - proceed to next step
            return {
                "status": "approved",
                "message": f"Human approved: {num_containers} containers to {destination}",
                "order_id": f"ORD-{num_containers}-HUMAN",
                "step": "scheduling",
            }
        else:
            # Human rejected - cancel the order
            return {
                "status": "rejected",
                "message": f"Human rejected: {num_containers} containers to {destination}",
                "order_id": f"ORD-{num_containers}-CANCELLED",
                "step": "cancelled",
            }


def create_multi_step_agent() -> LlmAgent:
    """
    Creates a multi-step shipping agent with enhanced workflow.
    
    Returns:
        LlmAgent: An agent configured for complex, multi-step shipping workflows
    """
    multi_step_agent = LlmAgent(
        name="multi_step_shipping_agent",
        model=Gemini(model=GEMINI_MODEL, retry_options=RETRY_CONFIG),
        instruction="""You are a multi-step shipping coordinator assistant.

    Workflow:
    1. Check order size and request approval if needed
    2. If approved, schedule the shipment
    3. Confirm scheduling and provide tracking details
    4. Handle each step clearly with status updates

    Always:
    - Use the place_shipping_order_with_tracking tool
    - Track the current step in your responses
    - Provide clear next steps or completion status
    """,
        tools=[FunctionTool(func=place_shipping_order_with_tracking)],
    )
    print(f"  - Multi-step agent '{multi_step_agent.name}' created with state tracking")
    return multi_step_agent


async def demo_resumable_workflows() -> None:
    """
    Demonstrates resumable workflows that maintain state across conversation breaks.
    """
    print("--- (3/3) Demonstrating Resumable Workflows ---")
    
    try:
        # Create a session service and runner for the multi-step agent
        session_service = InMemorySessionService()
        multi_step_agent = create_multi_step_agent()
        multi_step_app = App(
            name="multi_step_shipping_coordinator",
            root_agent=multi_step_agent,
            resumability_config=ResumabilityConfig(is_resumable=True),
        )
        
        multi_step_runner = Runner(
            app=multi_step_app,
            session_service=session_service,
        )

        query1 = "Ship 15 containers to Hamburg"
        session_id = f"resumable_{uuid.uuid4().hex[:8]}"
        
        print(f"\n📋 Testing resumable workflow with query: '{query1}'")
        
        # Create session
        await session_service.create_session(
            app_name="multi_step_shipping_coordinator",
            user_id="test_user",
            session_id=session_id
        )

        # Step 1: Initial request (will pause for approval)
        events1 = []
        async for event in multi_step_runner.run_async(
            user_id="test_user",
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=query1)])
        ):
            events1.append(event)

        # Check for approval request
        approval_info = check_for_approval(events1)
        
        if approval_info:
            print(f"⏸️  Workflow paused at step: approval_required")

            # Step 2: Resume with approval (simulating user returning later)
            print("\n🔄 Simulating user returning to approve...")
            
            events2 = []
            async for event in multi_step_runner.run_async(
                user_id="test_user",
                session_id=session_id,
                new_message=create_approval_response(approval_info, True),
                invocation_id=approval_info["invocation_id"],
            ):
                events2.append(event)

            print("\n✅ Workflow resumed and completed!")
            print_agent_response(events2)
        else:
            print_agent_response(events1)
        
        print("\n🎉 Resumable workflow test completed!")
        print("💡 Key insight: The agent remembered its state across the pause/resume cycle")
    except Exception as e:
        print(f"❌ Error in resumable workflow demo: {e}")
        # Continue with other demos even if this one fails


# --- Section 4: Production-Ready Example ---

def process_document(doc_type: str, sensitivity: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Process a document with approval for sensitive content.

    Args:
        doc_type: Type of document (contract, invoice, report)
        sensitivity: Sensitivity level (public, internal, confidential)
        tool_context: ADK tool context for requesting confirmation

    Returns:
        dict: Processing status and details
    """
    # Check if this is the first call or a resumed call
    if not tool_context.tool_confirmation:
        # First call - check if approval is needed
        if sensitivity.lower() == "confidential":
            # Confidential document - request approval
            tool_context.request_confirmation(
                hint=f"Confidential {doc_type} requires approval before processing.",
                payload={
                    "title": "Document Processing Approval",
                    "description": f"Approve processing of confidential {doc_type}"
                },
            )
            return {
                "status": "pending_approval",
                "message": f"Awaiting approval for {doc_type} document (confidential)",
                "doc_id": f"DOC-{doc_type.upper()}-{uuid.uuid4().hex[:6]}",
                "step": "approval_required",
            }
        else:
            # Non-confidential - process immediately
            return {
                "status": "processing",
                "message": f"Processing {doc_type} document ({sensitivity})",
                "doc_id": f"DOC-{doc_type.upper()}-{uuid.uuid4().hex[:6]}",
                "step": "analyzing",
            }
    else:
        # Resumed call - check the approval decision
        if tool_context.tool_confirmation.confirmed:
            # Human approved - process the document
            return {
                "status": "approved",
                "message": f"Human approved: Processing {doc_type} document",
                "doc_id": f"DOC-{doc_type.upper()}-{uuid.uuid4().hex[:6]}",
                "step": "processing_complete",
            }
        else:
            # Human rejected - cancel processing
            return {
                "status": "rejected",
                "message": f"Human rejected: {doc_type} document processing",
                "doc_id": f"DOC-{doc_type.upper()}-{uuid.uuid4().hex[:6]}",
                "step": "cancelled",
            }


def create_document_agent() -> LlmAgent:
    """
    Creates a document processing agent that combines MCP, long-running ops, and resumability.
    
    Returns:
        LlmAgent: A production-ready document processing agent
    """
    document_agent = LlmAgent(
        name="document_processor",
        model=Gemini(model=GEMINI_MODEL, retry_options=RETRY_CONFIG),
        instruction="""You are a document processing assistant with approval capabilities.

    Workflow:
    1. Analyze document type and sensitivity
    2. Request approval for confidential documents
    3. Process approved documents and provide status
    4. Handle rejections gracefully

    Always:
    - Use the process_document tool
    - Track processing steps clearly
    - Provide document IDs for reference
    """,
        tools=[FunctionTool(func=process_document)],
    )
    print(f"  - Document processing agent '{document_agent.name}' created with full capabilities")
    return document_agent


async def demo_production_example() -> None:
    """
    Demonstrates a production-ready agent combining all three advanced patterns.
    """
    print("--- Production-Ready Agent Demo ---")
    
    try:
        # Create session service and runner for document agent
        session_service = InMemorySessionService()
        document_agent = create_document_agent()
        document_app = App(
            name="document_processor",
            root_agent=document_agent,
            resumability_config=ResumabilityConfig(is_resumable=True),
        )
        
        document_runner = Runner(
            app=document_app,
            session_service=session_service,
        )

        # Test with a public document (no approval needed)
        print("🧪 Test 1: Public document (auto-approved)")
        query_content = types.Content(role="user", parts=[types.Part(text="Process invoice document (public sensitivity)")])
        session_id = f"doc_{uuid.uuid4().hex[:8]}"
        
        await session_service.create_session(
            app_name="document_processor", user_id="test_user", session_id=session_id
        )
        
        # Use the same approach as the MCP demo for consistency
        runner = InMemoryRunner(agent=document_agent)
        response = await runner.run_debug("Process invoice document (public sensitivity)")
        print("--- Document Processing Result ---")
        print(response)
        print("------------------------------")

        # Test with a confidential document (approval required)
        print("\n🧪 Test 2: Confidential document (approval required)")
        query_content2 = types.Content(role="user", parts=[types.Part(text="Process contract document (confidential sensitivity)")])
        session_id2 = f"doc_{uuid.uuid4().hex[:8]}"
        
        await session_service.create_session(
            app_name="document_processor", user_id="test_user", session_id=session_id2
        )
        
        # Initial call - should trigger approval request
        events = []
        async for event in document_runner.run_async(
            user_id="test_user",
            session_id=session_id2,
            new_message=query_content2
        ):
            events.append(event)

        # Check for approval request and provide approval
        approval_info = check_for_approval(events)
        if approval_info:
            print(f"  - Pausing for approval...")
            print(f"  - Human Decision: APPROVE ✅\n")
            
            # Resume with approval
            async for event in document_runner.run_async(
                user_id="test_user",
                session_id=session_id2,
                new_message=create_approval_response(approval_info, True),
                invocation_id=approval_info["invocation_id"],
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            print(f"Agent > {part.text}")

        print("\n🎉 Production-ready agent testing complete!")
        print("💡 Successfully demonstrated MCP, Long-Running Ops, and Resumable Workflows")
    except Exception as e:
        print(f"❌ Error in production example demo: {e}")
        # Continue with completion even if this demo fails


# --- Main Execution ---

async def main() -> None:
    """
    The main asynchronous entry point for running the advanced agent patterns demo.
    
    This function orchestrates the entire demonstration: setting up the environment,
    and executing examples of MCP integration, long-running operations, and resumable workflows.
    """
    print("--- Starting Advanced Agent Tool Patterns Demo ---")
    setup_environment()
    
    try:
        # Run each demo with individual error handling
        await demo_mcp_integration()
        
        # Add a small delay between demos to allow for cleanup
        await asyncio.sleep(0.5)
        
        await demo_long_running_operations()
        
        await asyncio.sleep(0.5)
        
        await demo_resumable_workflows()
        
        await asyncio.sleep(0.5)
        
        await demo_production_example()
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error in main demo: {e}")
    finally:
        # Clean up any remaining resources
        try:
            # Clean up MCP resources
            await cleanup_mcp_resources()
            # Give time for any cleanup operations
            await asyncio.sleep(0.2)
        except Exception:
            pass

    print("\n--- Advanced Agent Tool Patterns Demo Complete ---")


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