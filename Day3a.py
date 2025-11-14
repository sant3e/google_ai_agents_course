# -*- coding: utf-8 -*-
"""
Day 3A: Building Stateful Agents with Sessions.

This script provides a production-grade, educational example of how to build
stateful AI agents that can remember information across multiple turns of a
conversation using Google's Agent Development Kit (ADK).

Key Concepts Illustrated:
- Session Management: The core concept of a 'session' to track a single,
  continuous conversation.
- Stateful Agents: How to make an agent remember context and user inputs
  from previous turns.
- Agent Callbacks: Using `before_agent_callback` and `after_agent_callback`
  to load and save state at the beginning and end of each agent turn.
- State Serialization: Defining a structured `ConversationState` using
  Pydantic for reliable serialization and deserialization of session data.
- Session Persistence: Utilizing `InMemorySessionService` for basic,
  script-level state persistence and understanding how it can be replaced
  with more robust services like `SqliteSessionService`.

Prerequisites:
- google-adk
- python-dotenv
- pydantic
- A .env file in the root directory with a GOOGLE_API_KEY.
"""

from initial_setup import *
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types
from pydantic import BaseModel, Field

# --- State Management Definition ---

# Global session state storage to persist between callbacks
# This is a workaround for the ADK framework creating new session objects
_global_session_states = {}


class ConversationState(BaseModel):
    """
    A Pydantic model to define the structure of our agent's memory.

    Using a Pydantic model is a best practice for managing state because it
    provides clear, enforceable schema for the data you want to persist.
    This prevents bugs related to missing or malformed state data.

    Attributes:
        user_name: The name of the user, extracted from the conversation.
                   Defaults to None if not yet known.
        turn_count: A counter for the number of interactions in the session.
    """

    user_name: str | None = Field(
        default=None,
        description="The name of the user, learned from the conversation.",
    )
    turn_count: int = Field(
        default=0, description="The number of turns in the current conversation."
    )


# --- Agent Callback Implementations ---


def load_conversation_history(**kwargs) -> None:
    """
    Callback function executed *before* the agent processes a new message.

    This function is responsible for loading the `ConversationState` from the
    current session and attaching it to the agent's context. This makes the
    state available within the agent's instruction prompt.

    Args:
        kwargs: The keyword arguments provided by the ADK callback system.
    """
    context = kwargs.get('context') or kwargs.get('callback_context')
    session: Session = context.session
    print(f"  [Callback] 📥 Loading state for session: {session.id}")

    # Use our global state storage to get the state for this session
    global _global_session_states
    session_id = session.id

    # Check if we have state for this session in our global storage
    if session_id in _global_session_states:
        state_dict = _global_session_states[session_id]
        print(f"  [Debug] Load callback - found state in global storage: {state_dict}")
    else:
        # Try to get state from the session object
        state_dict = session.state if hasattr(session, 'state') else {}
        print(f"  [Debug] Load callback - using session state: {state_dict}")

        # If session state is empty, initialize with default values
        if not state_dict:
            state_dict = ConversationState().model_dump()
            print(f"  [Debug] Load callback - initialized with default state: {state_dict}")

    # Deserialize the dictionary into our Pydantic model.
    # This ensures the state is always structured correctly.
    state = ConversationState.model_validate(state_dict)
    print(f"  [Debug] Load callback - validated state: {state.model_dump_json()}")

    # Store in our global storage to ensure it persists
    _global_session_states[session_id] = state.model_dump()
    print(f"  [Debug] Load callback - stored state in global storage")

    # Store in kwargs for the agent to use during processing
    kwargs['memory'] = state
    print(f"  [Debug] Load callback - stored state in kwargs['memory']")

    # Try to store in context.data as well for compatibility
    if hasattr(context, 'data'):
        context.data["memory"] = state
        print(f"  [Debug] Load callback - stored state in context.data['memory']")
    else:
        try:
            context.data = {}
            context.data["memory"] = state
            print(f"  [Debug] Load callback - created context.data and stored state")
        except Exception as e:
            print(f"  [Debug] Load callback - failed to create context.data: {e}")

    # Update session state to include this conversation state
    session_state_dict = state.model_dump()
    if hasattr(session, 'state'):
        # Merge with existing state if needed
        existing_state = session.state if isinstance(session.state, dict) else {}
        existing_state.update(session_state_dict)
        session.state = existing_state
    else:
        session.state = session_state_dict

    print(f"  [Callback] State loaded: {state.model_dump_json()}")


def save_conversation_history(**kwargs) -> None:
    """
    Callback function executed *after* the agent has generated a response.

    This function is responsible for saving the (potentially updated)
    `ConversationState` back into the session, persisting it for the next turn.

    Args:
        kwargs: The keyword arguments provided by the ADK callback system.
    """
    context = kwargs.get('context') or kwargs.get('callback_context')
    session: Session = context.session
    print(f"  [Callback] 💾 Saving state for session: {session.id}")

    # Use our global state storage to get the current state
    global _global_session_states
    session_id = session.id

    # Try to get the state from kwargs first (where the agent might have modified it)
    state = kwargs.get('memory')

    # If not in kwargs, try to get it from context.data
    if not state and hasattr(context, 'data'):
        state = context.data.get("memory")
        print(f"  [Debug] Save callback - retrieved state from context.data: {state}")

    # If still not found, try to get it from our global storage
    if not state and session_id in _global_session_states:
        state_dict = _global_session_states[session_id]
        state = ConversationState.model_validate(state_dict)
        print(f"  [Debug] Save callback - retrieved state from global storage: {state.model_dump_json()}")

    if state:
        print(f"  [Debug] Save callback - final state to save: {state.model_dump_json()}")

        # Try multiple ways to get the agent's output
        agent_output = ""

        # Method 1: Check if event is in kwargs (most likely source during streaming)
        if 'event' in kwargs:
            event = kwargs['event']
            print(f"  [Debug] Save callback - found event in kwargs: {type(event)}")
            if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts') and event.content.parts:
                agent_output = event.content.parts[0].text
                print(f"  [Debug] Save callback - extracted agent_output from kwargs.event: {agent_output}")

        # Method 2: Check if event is in context
        if not agent_output and hasattr(context, 'event') and context.event:
            event = context.event
            print(f"  [Debug] Save callback - found event in context: {type(event)}")
            if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts') and event.content.parts:
                agent_output = event.content.parts[0].text
                print(f"  [Debug] Save callback - extracted agent_output from context.event: {agent_output}")

        # Method 3: Check if response exists in context (fallback)
        if not agent_output and hasattr(context, 'response') and context.response:
            response = context.response
            print(f"  [Debug] Save callback - found response in context: {type(response)}")
            if hasattr(response, 'text'):
                agent_output = response.text
                print(f"  [Debug] Save callback - extracted agent_output from context.response: {agent_output}")

        # Method 4: Check in kwargs (fallback)
        if not agent_output and 'response' in kwargs:
            response = kwargs['response']
            print(f"  [Debug] Save callback - found response in kwargs: {type(response)}")
            if hasattr(response, 'text'):
                agent_output = response.text
                print(f"  [Debug] Save callback - extracted agent_output from kwargs.response: {agent_output}")

        # Method 5: Check for other possible attributes in context that might contain the output
        if not agent_output and hasattr(context, 'result'):
            result = context.result
            if hasattr(result, 'text'):
                agent_output = result.text
                print(f"  [Debug] Save callback - extracted agent_output from context.result: {agent_output}")
            elif hasattr(result, 'content') and result.content and hasattr(result.content, 'parts') and result.content.parts:
                agent_output = result.content.parts[0].text
                print(f"  [Debug] Save callback - extracted agent_output from context.result.content: {agent_output}")

        print(f"  [Debug] Save callback - final agent_output: '{agent_output}'")

        # Extract the user's name if present
        if "EXTRACTED_NAME:" in agent_output and agent_output:
            extracted_name = agent_output.split("EXTRACTED_NAME:")[1].strip().split()[0]  # Extract first word after EXTRACTED_NAME
            if extracted_name and state.user_name is None:
                state.user_name = extracted_name
                print(f"  [Callback] ✨ User name '{extracted_name}' extracted and saved.")
                print(f"  [Debug] Save callback - state after name extraction: {state.model_dump_json()}")

        # Increment the turn count
        state.turn_count += 1
        print(f"  [Debug] Save callback - state after incrementing turn count: {state.model_dump_json()}")

        # Serialize the Pydantic model back to a dictionary
        state_dict = state.model_dump()
        print(f"  [Debug] Save callback - saving state_dict: {state_dict}")

        # Store in our global storage to ensure it persists
        _global_session_states[session_id] = state_dict
        print(f"  [Debug] Save callback - stored state in global storage")

        # Also try to update the session object for completeness
        if hasattr(session, 'set_state'):
            session.set_state(state_dict)
        else:
            session.state = state_dict

        print(f"  [Callback] State saved: {state.model_dump_json()}")
    else:
        print("  [Callback] ⚠️ No state found to save.")


# --- Agent Definition ---


def create_stateful_greeter_agent() -> Agent:
    """
    Defines and constructs a stateful agent that remembers the user's name.

    This agent uses callbacks to load and save its state, and its instruction
    prompt is templated to include the `ConversationState` (as `{memory}`).

    Returns:
        An instance of a configured ADK Agent with stateful capabilities.
    """
    # Updated instruction to not rely on {memory} template since it's causing KeyError
    # Instead, the memory will be managed through callbacks
    greeter_instruction = """
You are a friendly and polite greeter. Your goal is to learn the user's name
and remember it for the rest of the conversation.

**Your Task:**
1.  If you learn the user's name, you should **include their name in your response** in the
    following format: `EXTRACTED_NAME: [user's name]`.
    For example: "Nice to meet you, Bob! EXTRACTED_NAME: Bob"

2.  If you already know the user's name, use it to greet them personally in
    your response. Do NOT use the `EXTRACTED_NAME` format again.

3.  If the user asks a question, answer it politely.
"""

    greeter_agent = Agent(
        name="greeter_agent",
        model=GEMINI_MODEL,
        instruction=greeter_instruction,
        before_agent_callback=load_conversation_history,
        after_agent_callback=save_conversation_history,
    )
    print(f"🤖 Agent '{greeter_agent.name}' created with stateful callbacks.")
    return greeter_agent


# --- Main Demonstration Logic ---


async def run_conversation_turn(
    runner: Runner, session_id: str, user_id: str, query: str
) -> None:
    """
    Helper function to run a single turn of conversation and print the output.

    Args:
        runner: The Runner instance to execute the query.
        session_id: The ID of the current conversation session.
        user_id: The ID of the user.
        query: The user's input for this turn.
    """
    print("-" * 60)
    print(f"💬 User: '{query}'")
    print("🧠 Agent is thinking...")

    # Convert the query string to the ADK Content format
    query_content = types.Content(role="user", parts=[types.Part(text=query)])

    # Run the agent with streaming to ensure callbacks receive the events properly
    response_text = ""
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=query_content
    ):
        # Check if the event contains valid content
        if event.content and event.content.parts:
            # Filter out empty or "None" responses before printing
            if (
                hasattr(event.content.parts[0], 'text')
                and event.content.parts[0].text != "None"
                and event.content.parts[0].text
            ):
                response_text = event.content.parts[0].text
                print(f"greeter_agent > {response_text}")

    print("\n✅ Agent execution complete.")
    print("--- Agent's Final Response ---")
    print(response_text)
    print("------------------------------\n")

    # After getting the response, manually check if it contains name extraction
    # and update the global session state accordingly
    if "EXTRACTED_NAME:" in response_text:
        extracted_name = response_text.split("EXTRACTED_NAME:")[1].strip().split()[0]  # Extract first word after EXTRACTED_NAME
        print(f"  [Manual Update] Extracting name '{extracted_name}' from agent response")

        # Update the session state manually
        global _global_session_states
        if session_id in _global_session_states:
            current_state = _global_session_states[session_id]
            if current_state.get('user_name') is None:
                current_state['user_name'] = extracted_name
                print(f"  [Manual Update] Updated session state with user name: {extracted_name}")


async def main() -> None:
    """
    The main asynchronous entry point for the stateful agent demonstration.

    This function orchestrates the entire workflow:
    1. Sets up the environment.
    2. Creates the stateful agent.
    3. Initializes a `Runner` with an `InMemorySessionService` to manage state.
    4. Simulates a multi-turn conversation to show the agent remembering context.
    """
    print("--- Starting Stateful Agent (Sessions) Demonstration ---")
    setup_environment()

    # Create the agent with our defined callbacks.
    root_agent = create_stateful_greeter_agent()

    # The SessionService is the component responsible for storing and retrieving
    # session data. InMemorySessionService is perfect for simple scripts, as it
    # holds all data in memory. For production, you would use a persistent
    # service like `SqliteSessionService` or a custom database integration.
    session_service = InMemorySessionService()
    print(f"🗄️  Using session service: {type(session_service).__name__}")

    # The Runner orchestrates the agent execution and session management.
    print(f"  [Debug] Creating Runner with agent='{root_agent.name}', session_service='{type(session_service).__name__}'")
    runner = Runner(agent=root_agent, app_name="stateful_app", session_service=session_service)
    print("🏃 Runner initialized. Ready to process queries.")

    # Define a unique ID for this conversation. In a real application, this
    # would be managed per-user or per-conversation.
    session_id = "user-session-12345"
    user_id = "test-user"

    # Manually create the session before the first turn.
    print(f"  [Debug] Attempting to create session with app_name='stateful_app', user_id='{user_id}', session_id='{session_id}'")
    try:
        await session_service.create_session(
            app_name="stateful_app", user_id=user_id, session_id=session_id
        )
        print(f"  [Debug] Session created successfully")
    except Exception as e:
        print(f"  [Debug] Error creating session: {e}")
        raise
    print(f"🤝 Session '{session_id}' created for user '{user_id}'.")

    # --- Run the Conversational Flow ---
    await run_conversation_turn(
        runner, session_id, user_id, "Hello, I'm new here."
    )
    await run_conversation_turn(
        runner, session_id, user_id, "My name is Alex."
    )
    await run_conversation_turn(
        runner, session_id, user_id, "What's the weather like today?"
    )
    await run_conversation_turn(
        runner,
        session_id,
        user_id,
        "Do you remember my name?",
    )

    print("--- Stateful Agent (Sessions) Demonstration Complete ---")


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
#        python Day3a.py
#
# ==============================================================================

if __name__ == "__main__":
    # This block handles running the async main function in different environments.
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
