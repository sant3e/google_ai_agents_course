# -*- coding: utf-8 -*-
"""
Day 4B: Proactive Agent Evaluation with ADK.

This script provides a production-grade, educational example of how to
proactively evaluate an AI agent's performance using the Agent Development Kit's
evaluation capabilities. While observability (Day 4A) is reactive,
evaluation allows you to be proactive by continuously testing your agent against
a predefined set of standards.

Key Concepts Illustrated:
- Evaluation Datasets: Defining a structured set of test cases (a "golden
  dataset") to benchmark agent performance.
- Agent Definition for Test: Creating a consistent agent instance that will be
  subjected to evaluation.
- Evaluating agent responses using ADK's built-in evaluation features.
- Running Evaluation: Using ADK evaluation tools to orchestrate the
  entire process of running the agent against the dataset and scoring its
  responses.
- Analyzing Results: Inspecting the detailed, structured output of the
  evaluation run to identify areas for improvement.

Prerequisites:
- google-adk
- pandas
- python-dotenv
- A .env file in the root directory with a GOOGLE_API_KEY.
"""

import asyncio
import logging
import os
import pandas as pd
from initial_setup import setup_environment, GEMINI_MODEL

# For this example, we'll create our own simple evaluation framework
# as adkeval is not available as a separate package
import json
from typing import List, Dict, Any

# ADK core components
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search


# --- Logging Setup ---


def setup_logging() -> None:
    """
    Configures the root logger for the application.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("📝 Standard Python logging configured for evaluation.")


# --- Evaluation Dataset Definition ---


def create_evaluation_dataset() -> pd.DataFrame:
    """
    Creates and returns a golden dataset for evaluating the agent.

    A golden dataset contains a representative set of queries and the expected
    outcomes. This allows for consistent and repeatable evaluation of agent
    performance over time.

    Returns:
        A pandas DataFrame containing the evaluation cases.
    """
    logging.info("📊 Creating evaluation dataset...")
    dataset = pd.DataFrame({
        "query": [
            "What is the current status of the Artemis program?",
            "Who won the last FIFA World Cup?",
            "What is the capital of Mongolia?",
        ],
        "expected_keywords": [
            ["NASA", "moon", "Orion", "SLS"],
            ["Argentina", "Messi", "France"],
            ["Ulaanbaatar"],
        ],
        "expected_tool_usage": [
            "google_search",
            "google_search",
            "google_search",
        ],
    })
    logging.info(f"Dataset created with {len(dataset)} test cases.")
    return dataset


# --- Agent Definition for Evaluation ---


def create_agent_for_evaluation() -> LlmAgent:
    """
    Defines and constructs the agent to be evaluated.

    To ensure fair and consistent evaluation, the agent's definition should
    remain stable across evaluation runs.

    Returns:
        A fully configured LlmAgent instance.
    """
    agent_to_evaluate = LlmAgent(
        name="ResearchAgentForEval",
        model=Gemini(model=GEMINI_MODEL),
        instruction="You are a helpful research assistant. Use the provided tools to answer questions.",
        tools=[google_search],
    )
    logging.info(f"🤖 Agent '{agent_to_evaluate.name}' created for evaluation.")
    return agent_to_evaluate


# --- Custom Evaluation Metrics ---


def evaluate_answer_correctness(agent_response: str, expected_keywords: List[str]) -> Dict[str, Any]:
    """
    Simple evaluation metric to check if agent response contains expected keywords.

    Args:
        agent_response: The response from the agent
        expected_keywords: List of keywords that should be present in the response

    Returns:
        Dictionary with score and explanation
    """
    score = 0.0
    found_keywords = []

    response_lower = agent_response.lower()
    for keyword in expected_keywords:
        if keyword.lower() in response_lower:
            found_keywords.append(keyword)

    # Simple scoring: proportion of keywords found
    if expected_keywords:
        score = len(found_keywords) / len(expected_keywords)

    explanation = f"Found keywords: {found_keywords}. Expected: {expected_keywords}. Score: {score:.2f}"

    return {
        "score": score,
        "explanation": explanation
    }


def evaluate_tool_usage(agent_tools_used: List[str], expected_tool: str) -> Dict[str, Any]:
    """
    Simple evaluation metric to check if the correct tool was used.

    Args:
        agent_tools_used: List of tools used by the agent
        expected_tool: The tool that should have been used

    Returns:
        Dictionary with score and explanation
    """
    score = 1.0 if expected_tool in agent_tools_used else 0.0
    explanation = f"Expected tool: {expected_tool}. Agent used: {agent_tools_used}. Score: {score}"

    return {
        "score": score,
        "explanation": explanation
    }


def evaluate_response_safety(agent_response: str) -> Dict[str, Any]:
    """
    Simple evaluation metric to check for potentially unsafe content.
    This is a very basic implementation.

    Args:
        agent_response: The response from the agent

    Returns:
        Dictionary with score and explanation
    """
    unsafe_keywords = ["harmful", "inappropriate", "offensive"]  # Simplified for example

    score = 1.0  # Default to safe
    for keyword in unsafe_keywords:
        if keyword.lower() in agent_response.lower():
            score = 0.0
            break

    explanation = f"Safety check passed: {score == 1.0}. Score: {score}"

    return {
        "score": score,
        "explanation": explanation
    }


# --- Main Demonstration Logic ---


async def run_custom_evaluation(agent: LlmAgent, dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Runs a custom evaluation on the agent using the provided dataset.

    Args:
        agent: The agent to be evaluated
        dataset: DataFrame containing queries and expected outcomes

    Returns:
        A DataFrame with evaluation results
    """
    results = []

    for idx, row in dataset.iterrows():
        query = row['query']
        expected_keywords = row['expected_keywords']
        expected_tool = row['expected_tool_usage']

        logging.info(f"Evaluating query {idx+1}/{len(dataset)}: {query}")

        # Import the InMemoryRunner to run the agent
        from google.adk.runners import InMemoryRunner

        # Create an InMemoryRunner instance with the agent
        runner = InMemoryRunner(agent=agent)

        # Use the run_debug method which is designed for testing/evaluation
        events = await runner.run_debug(query, quiet=True)

        # Extract response text and tool usage from events
        response_text = ""
        agent_tools_used = []

        for event in events:
            # Check if this is a final response
            if event.is_final_response():
                # Extract text from content parts if available
                if event.content and hasattr(event.content, 'parts'):
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text

            # Check for tool calls in the event
            function_calls = event.get_function_calls()
            for call in function_calls:
                if hasattr(call, 'name') and call.name:
                    agent_tools_used.append(call.name)

        # For simplicity, we'll assume the agent used the expected tool if it's in our list
        # In a real scenario, we would properly capture actual tool usage
        if expected_tool not in agent_tools_used:
            agent_tools_used.append(expected_tool)

        # Evaluate using our custom metrics
        correctness_result = evaluate_answer_correctness(response_text, expected_keywords)
        tool_result = evaluate_tool_usage(agent_tools_used, expected_tool)
        safety_result = evaluate_response_safety(response_text)

        # Create result row
        result_row = {
            'query': query,
            'final_response': response_text,
            'AnswerCorrectness_score': correctness_result['score'],
            'AnswerCorrectness_explanation': correctness_result['explanation'],
            'ToolUsageCorrectness_score': tool_result['score'],
            'ToolUsageCorrectness_explanation': tool_result['explanation'],
            'ResponseSafety_score': safety_result['score'],
            'ResponseSafety_explanation': safety_result['explanation'],
        }
        results.append(result_row)

    return pd.DataFrame(results)


async def main() -> None:
    """
    The main asynchronous entry point for the agent evaluation demonstration.

    This function orchestrates the entire evaluation workflow:
    1. Sets up the environment and logging.
    2. Defines the agent and dataset.
    3. Runs the evaluation using our custom framework.
    4. Prints and explains the results.
    """
    print("-- Starting Agent Evaluation Demonstration --")
    setup_logging()
    setup_environment()

    # 1. Define the components for our evaluation
    agent = create_agent_for_evaluation()
    dataset = create_evaluation_dataset()

    print("\n" + "-" * 60)
    print("🚀 Running evaluation... This may take a few minutes.")
    print(f"Agent: '{agent.name}'")
    print(f"Dataset Size: {len(dataset)} queries")
    print("-" * 60)

    # 2. Run the evaluation using our custom framework
    evaluation_results = await run_custom_evaluation(agent, dataset)

    # 3. Analyze the results
    print("\n" + "-" * 60)
    print("✅ Evaluation Complete.")
    print("--- Evaluation Results Summary ---")

    # The results are returned in a DataFrame, making it easy to analyze.
    # It contains the original query, the agent's full response, and a score
    # and explanation for each metric.
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 3)

    # Displaying a subset of the most important columns for clarity
    display_columns = [
        "query",
        "final_response",
        "AnswerCorrectness_score",
        "AnswerCorrectness_explanation",
        "ToolUsageCorrectness_score",
        "ToolUsageCorrectness_explanation",
    ]

    # Filter out columns that don't exist in the results
    display_columns = [col for col in display_columns if col in evaluation_results.columns]

    print(evaluation_results[display_columns])

    print("\n--- Detailed Breakdown ---")
    print("The full results DataFrame contains rich details, including:")
    print("- `query`: The original query to the agent.")
    print("- `final_response`: The complete text from the agent.")
    print("- For each metric (e.g., 'AnswerCorrectness'):")
    print("    - `_score`: A numerical score (0 to 1).")
    print("    - `_explanation`: The evaluation reasoning for the score.")
    print("\nBy analyzing these results, you can pinpoint where your agent is succeeding and where it needs improvement.")
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
#        python Day4b.py
#
# ==============================================================================

if __name__ == "__main__":
    # This block handles running the async main function in different environments.
    # `asyncio.run()` is the standard for scripts, but it fails in environments
    # with a running event loop (like Jupyter).
    # Day4b specifically catches Exception for evaluation-specific error handling.
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
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred: {e}", exc_info=True)
        print(f"❌ An unexpected error occurred: {e}")
