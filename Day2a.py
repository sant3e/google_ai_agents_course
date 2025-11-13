# -*- coding: utf-8 -*-
"""
Day 2A: Building Agents with Custom Tools.

This script provides an educational walkthrough on creating and using custom tools
with Google's Agent Development Kit (ADK). It demonstrates how to extend an agent's
capabilities beyond its built-in knowledge by giving it access to Python functions,
other agents, and a code executor for reliable calculations.

Key Concepts Illustrated:
- LlmAgent: An advanced agent class with more configuration options.
- FunctionTool: Turning any Python function into a callable tool for an agent.
- AgentTool: Using a specialized agent as a tool for another agent.
- BuiltInCodeExecutor: Offloading calculations to a secure code execution
  environment to improve reliability and precision.
- Retry Configuration: Making agent interactions with external services more robust.

Prerequisites:
- google-adk
- python-dotenv
- A .env file in the root directory with a GOOGLE_API_KEY.
"""

from initial_setup import GEMINI_MODEL, setup_environment, asyncio, sys
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool
from google.adk.code_executors import BuiltInCodeExecutor

# --- Constants and Configuration ---

# Configure retry options for handling transient network errors, making the agent more robust.
RETRY_CONFIG = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on common server/rate limit errors
)


# --- Tool Definitions ---


def get_fee_for_payment_method(method: str) -> dict:
    """
    Looks up the transaction fee percentage for a given payment method.

    This tool simulates looking up a company's internal fee structure based on the
    name of the payment method provided by the user.

    Args:
        method: The name of the payment method (e.g., "platinum credit card").

    Returns:
        A dictionary with status and fee information.
        Success: {"status": "success", "fee_percentage": 0.02}
        Error: {"status": "error", "error_message": "Payment method not found"}
    """
    fee_database = {
        "platinum credit card": 0.02,  # 2%
        "gold debit card": 0.035,  # 3.5%
        "bank transfer": 0.01,  # 1%
        "paypal": 0.025,  # 2.5%
        "wire transfer": 0.015,  # 1.5%
    }
    fee = fee_database.get(method.lower())
    if fee is not None:
        return {"status": "success", "fee_percentage": fee}
    else:
        return {
            "status": "error",
            "error_message": f"Payment method '{method}' not found. Available methods: {list(fee_database.keys())}",
        }


def get_exchange_rate(base_currency: str, target_currency: str) -> dict:
    """
    Looks up and returns the exchange rate between two currencies.

    This tool simulates a live exchange rate API call.

    Args:
        base_currency: The ISO 4217 code of the currency to convert from (e.g., "USD").
        target_currency: The ISO 4217 code of the currency to convert to (e.g., "EUR").

    Returns:
        A dictionary with status and rate information.
        Success: {"status": "success", "rate": 0.93}
        Error: {"status": "error", "error_message": "Unsupported currency pair"}
    """
    rate_database = {
        "usd": {"eur": 0.93, "jpy": 157.50, "inr": 83.58, "gbp": 0.79, "cad": 1.36},
        "eur": {"usd": 1.08, "gbp": 0.85, "jpy": 169.89},
    }
    base = base_currency.lower()
    target = target_currency.lower()
    rate = rate_database.get(base, {}).get(target)
    if rate is not None:
        return {"status": "success", "rate": rate}
    else:
        return {
            "status": "error",
            "error_message": f"Unsupported currency pair: {base_currency.upper()}/{target_currency.upper()}",
        }


# --- Agent Definitions ---


def define_simple_currency_agent() -> LlmAgent:
    """
    Defines an agent that uses function tools but performs math itself.

    This agent demonstrates the basics of using FunctionTools but highlights a
    common weakness: relying on the LLM for precise mathematical calculations.

    Returns:
        An LlmAgent configured with currency tools.
    """
    currency_agent = LlmAgent(
        name="currency_agent",
        model=Gemini(model=GEMINI_MODEL, retry_options=RETRY_CONFIG),
        instruction="""You are a smart currency conversion assistant.
        1. Use `get_fee_for_payment_method()` to find transaction fees.
        2. Use `get_exchange_rate()` to get currency conversion rates.
        3. Check the "status" field in each tool's response for errors.
        4. Calculate the final amount after fees based on the tool outputs.
        5. Provide a clear breakdown of the calculation.
        If any tool returns an error, explain the issue clearly.
        """,
        tools=[get_fee_for_payment_method, get_exchange_rate],
    )
    print("  - Simple 'currency_agent' created.")
    return currency_agent


def define_enhanced_currency_agent() -> LlmAgent:
    """
    Defines an advanced agent that delegates calculations to a specialist.

    This system demonstrates a more robust architecture:
    1. A `CalculationAgent` is created with a `BuiltInCodeExecutor`. Its sole
       purpose is to translate calculation requests into Python code and execute it.
    2. The `enhanced_currency_agent` uses the standard function tools for data
       retrieval and the `CalculationAgent` (as an `AgentTool`) for math.

    Returns:
        The top-level `enhanced_currency_agent`.
    """
    # This specialist agent generates and executes Python code for calculations.
    calculation_agent = LlmAgent(
        name="CalculationAgent",
        model=Gemini(model=GEMINI_MODEL, retry_options=RETRY_CONFIG),
        instruction="""You are a specialized calculator that ONLY responds with Python code.
        Your task is to take a calculation request and translate it into a single block of Python code that calculates the answer.
        The Python code MUST print the final result to stdout.
        You are PROHIBITED from performing the calculation yourself.
        Your output MUST be ONLY a Python code block.
        """,
        code_executor=BuiltInCodeExecutor(),
    )
    print("  - Specialist 'CalculationAgent' with Code Executor created.")

    # This agent orchestrates data retrieval and delegates math to the specialist.
    enhanced_currency_agent = LlmAgent(
        name="enhanced_currency_agent",
        model=Gemini(model=GEMINI_MODEL, retry_options=RETRY_CONFIG),
        instruction="""You are a smart currency conversion assistant. You must strictly follow these steps:
        1. Get Transaction Fee: Use `get_fee_for_payment_method()`.
        2. Get Exchange Rate: Use `get_exchange_rate()`.
        3. Error Check: If any tool call returns a status of "error", stop and explain the issue.
        4. Calculate Final Amount (CRITICAL): You are strictly prohibited from performing any arithmetic yourself.
           You MUST use the `CalculationAgent` tool to generate and execute Python code for the final calculation.
        5. Provide a detailed breakdown of the fee, rate, and final amount.
        """,
        tools=[
            get_fee_for_payment_method,
            get_exchange_rate,
            AgentTool(agent=calculation_agent),  # Use the specialist agent as a tool
        ],
    )
    print("  - 'enhanced_currency_agent' created, using CalculationAgent as a tool.")
    return enhanced_currency_agent


# --- Demonstration Logic ---


def _print_demonstration_divider():
    """Prints a clear visual divider between demonstrations."""
    print("\n" + "=" * 80 + "\n")


async def demo_simple_currency_agent():
    """
    Demonstrates the simple currency agent that performs math itself.
    Observe how the LLM might make small errors or inconsistencies in its calculation.
    """
    print("--- (1/2) Demonstrating Simple Currency Agent (LLM does the math) ---")
    agent = define_simple_currency_agent()
    runner = InMemoryRunner(agent=agent)
    query = "I want to convert 500 US Dollars to Euros using my Platinum Credit Card. How much will I receive?"
    print(f"\n💵 Query: '{query}'")
    print("🧠 Agent is thinking...")
    response = await runner.run_debug(query)
    print("\n💰 Conversion Result:")
    print(response)


async def demo_enhanced_currency_agent():
    """
    Demonstrates the enhanced agent that delegates math to a code executor.
    The result is more reliable and the calculation steps are verifiable.
    """
    print("--- (2/2) Demonstrating Enhanced Currency Agent (delegates math to code) ---")
    agent = define_enhanced_currency_agent()
    runner = InMemoryRunner(agent=agent)
    query = "Convert 1,250 USD to INR using a Bank Transfer. Show me the precise calculation."
    print(f"\n💵 Query: '{query}'")
    print("🧠 Agent is thinking...")
    response = await runner.run_debug(query)
    print("\n💰 Enhanced Conversion Result:")
    print(response)


async def main() -> None:
    """
    The main asynchronous entry point for demonstrating custom ADK tools.
    """
    print("--- Starting Custom Tool Demonstrations ---")
    setup_environment()

    await demo_simple_currency_agent()
    _print_demonstration_divider()
    await demo_enhanced_currency_agent()

    print("\n--- All Custom Tool Demonstrations Complete ---")


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
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
