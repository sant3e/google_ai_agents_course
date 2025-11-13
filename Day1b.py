# -*- coding: utf-8 -*-
"""
Day 1B: Advanced Agent Architectures with Google's ADK.

This script explores advanced agent architectures using Google's Agent Development Kit (ADK),
demonstrating how to build multi-agent systems with specialized roles and different
workflow patterns: sequential, parallel, and loop-based.

It builds upon the foundational concepts introduced in Day 1A, showcasing how to
orchestrate multiple agents to tackle more complex tasks efficiently and robustly.

Key Concepts Illustrated:
- Multi-Agent Systems: Breaking down complex tasks into specialized agent roles.
- SequentialAgent: Orchestrating agents in a fixed, ordered pipeline.
- ParallelAgent: Executing independent agent tasks concurrently.
- LoopAgent: Implementing iterative refinement workflows with feedback loops.
- AgentTool & FunctionTool: Using agents and Python functions as tools within other agents.
- State Management: Information flow between agents using `output_key` and instruction templating.

Prerequisites:
- google-adk
- python-dotenv
- A .env file in the root directory with a GOOGLE_API_KEY.
"""

from initial_setup import GEMINI_MODEL, setup_environment
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search

# --- Constants and Configuration ---

MAX_LOOP_ITERATIONS = 3
"""Maximum iterations for the LoopAgent to prevent infinite loops."""


# --- Section 1: Multi-Agent System (Coordinator, Research, Summarizer) ---


def define_multi_agent_research_system() -> Agent:
    """
    Defines a multi-agent system for research and summarization.

    This system consists of three agents:
    1. ResearchAgent: Uses Google Search to find information.
    2. SummarizerAgent: Summarizes the findings from the ResearchAgent.
    3. ResearchCoordinator: Orchestrates the workflow by calling the other two
       agents as tools in a specific sequence defined by its instruction.

    Returns:
        Agent: The root ResearchCoordinator agent of this multi-agent system.
    """
    # Research Agent: Its job is to use the google_search tool and present findings.
    research_agent = Agent(
        name="ResearchAgent",
        model=GEMINI_MODEL,
        instruction="""You are a specialized research agent. Your only job is to use the
        google_search tool to find 2-3 pieces of relevant information on the given topic and present the findings with citations.

        Important:
        - Focus on finding recent, reliable information.
        - Include sources/citations when possible.
        - Keep your findings concise but comprehensive.
        - Do not summarize - just present the research findings.
        """,
        tools=[google_search],
        output_key="research_findings",  # The result will be stored with this key
    )
    print(f"  - ResearchAgent created. Output: '{research_agent.output_key}'")

    # Summarizer Agent: Its job is to summarize the text it receives.
    summarizer_agent = Agent(
        name="SummarizerAgent",
        model=GEMINI_MODEL,
        instruction="""Read the provided research findings: {research_findings}

        Create a concise summary as a bulleted list with 3-5 key points.

        Guidelines:
        - Extract the most important information.
        - Use clear, concise bullet points.
        - Maintain accuracy to the source material.
        - Focus on actionable insights.
        """,
        output_key="final_summary",
    )
    print(f"  - SummarizerAgent created. Input: '{{research_findings}}', Output: '{summarizer_agent.output_key}'")

    # Root Coordinator: Orchestrates the workflow by calling the sub-agents as tools.
    research_coordinator = Agent(
        name="ResearchCoordinator",
        model=GEMINI_MODEL,
        instruction="""You are a research coordinator. Your goal is to answer the user's query by orchestrating a workflow.

        Follow these steps EXACTLY in order:
        1. First, you MUST call the `ResearchAgent` tool to find relevant information on the topic provided by the user.
        2. Next, after receiving the research findings, you MUST call the `SummarizerAgent` tool to create a concise summary.
        3. Finally, present the final summary clearly to the user as your response.

        Important:
        - Do not skip any steps.
        - Wait for each agent to complete before calling the next.
        - Present the final summary as your answer to the user.
        """,
        # We wrap the sub-agents in `AgentTool` to make them callable tools for the root agent.
        tools=[
            AgentTool(research_agent),
            AgentTool(summarizer_agent)
        ],
    )
    print(f"  - ResearchCoordinator created with tools: {[tool.agent.name for tool in research_coordinator.tools]}")
    return research_coordinator


# --- Section 2: Sequential Workflows (Blog Post Pipeline) ---


def define_sequential_blog_pipeline() -> SequentialAgent:
    """
    Defines a sequential agent pipeline for generating a blog post.

    This pipeline demonstrates how `SequentialAgent` ensures a fixed order of execution,
    where the output of one agent becomes the input for the next.

    The pipeline consists of:
    1. OutlineAgent: Creates a blog outline.
    2. WriterAgent: Writes a draft based on the outline.
    3. EditorAgent: Edits and polishes the draft.

    Returns:
        SequentialAgent: The sequential agent representing the blog post pipeline.
    """
    # Outline Agent: Creates the initial blog post outline.
    outline_agent = Agent(
        name="OutlineAgent",
        model=GEMINI_MODEL,
        instruction="""Create a blog outline for the given topic with:
        1. A catchy headline.
        2. An introduction hook.
        3. 3-5 main sections with 2-3 bullet points for each.
        4. A concluding thought.

        Make the outline engaging and well-structured.
        """,
        output_key="blog_outline",
    )
    print(f"  - OutlineAgent created. Output: '{outline_agent.output_key}'")

    # Writer Agent: Writes the full blog post based on the outline.
    writer_agent = Agent(
        name="WriterAgent",
        model=GEMINI_MODEL,
        instruction="""Following this outline strictly: {blog_outline}

        Write a brief, 200 to 300-word blog post with an engaging and informative tone.

        Guidelines:
        - Follow the outline structure exactly.
        - Keep within the word count.
        - Use clear, accessible language.
        - Include a compelling introduction and conclusion.
        """,
        output_key="blog_draft",
    )
    print(f"  - WriterAgent created. Input: '{{blog_outline}}', Output: '{writer_agent.output_key}'")

    # Editor Agent: Edits and polishes the draft from the writer agent.
    editor_agent = Agent(
        name="EditorAgent",
        model=GEMINI_MODEL,
        instruction="""Edit this draft: {blog_draft}

        Your task is to polish the text by:
        - Fixing any grammatical errors.
        - Improving the flow and sentence structure.
        - Enhancing overall clarity.
        - Ensuring consistent tone.
        - Maintaining the original meaning and word count.
        """,
        output_key="final_blog",
    )
    print(f"  - EditorAgent created. Input: '{{blog_draft}}', Output: '{editor_agent.output_key}'")

    # Create a sequential agent that runs the agents in order
    blog_pipeline = SequentialAgent(
        name="BlogPipeline",
        sub_agents=[outline_agent, writer_agent, editor_agent],
    )
    print(f"  - SequentialAgent 'BlogPipeline' created with sub-agents: {[a.name for a in blog_pipeline.sub_agents]}")
    return blog_pipeline


# --- Section 3: Parallel Workflows (Multi-Domain Research Team) ---


def define_parallel_research_system() -> SequentialAgent:
    """
    Defines a system that performs parallel research followed by aggregation.

    This system showcases `ParallelAgent` for concurrent execution of independent
    research tasks, followed by a `SequentialAgent` to aggregate their results.

    The system includes:
    - TechResearcher, HealthResearcher, FinanceResearcher: Run in parallel.
    - AggregatorAgent: Combines findings from all researchers.

    Returns:
        SequentialAgent: The root sequential agent orchestrating parallel research and aggregation.
    """
    # Tech Researcher: Focuses on AI and ML trends.
    tech_researcher = Agent(
        name="TechResearcher",
        model=GEMINI_MODEL,
        instruction="""Research the latest AI/ML trends. Include:
        - 3 key developments.
        - The main companies involved.
        - The potential impact.

        Keep the report very concise (around 100 words).
        """,
        tools=[google_search],
        output_key="tech_research",
    )
    print(f"  - TechResearcher created. Output: '{tech_researcher.output_key}'")

    # Health Researcher: Focuses on medical breakthroughs.
    health_researcher = Agent(
        name="HealthResearcher",
        model=GEMINI_MODEL,
        instruction="""Research recent medical breakthroughs. Include:
        - 3 significant advances.
        - Their practical applications.
        - Estimated timelines.

        Keep the report concise (around 100 words).
        """,
        tools=[google_search],
        output_key="health_research",
    )
    print(f"  - HealthResearcher created. Output: '{health_researcher.output_key}'")

    # Finance Researcher: Focuses on fintech trends.
    finance_researcher = Agent(
        name="FinanceResearcher",
        model=GEMINI_MODEL,
        instruction="""Research current fintech trends. Include:
        - 3 key trends.
        - Their market implications.
        - The future outlook.

        Keep the report concise (around 100 words).
        """,
        tools=[google_search],
        output_key="finance_research",
    )
    print(f"  - FinanceResearcher created. Output: '{finance_researcher.output_key}'")

    # The AggregatorAgent runs *after* the parallel step to synthesize the results.
    aggregator_agent = Agent(
        name="AggregatorAgent",
        model=GEMINI_MODEL,
        instruction="""Combine these three research findings into a single executive summary:

        **Technology Trends:**
        {tech_research}

        **Health Breakthroughs:**
        {health_research}

        **Finance Innovations:**
        {finance_research}

        Your summary should:
        - Highlight common themes across domains.
        - Identify surprising connections.
        - Present the most important key takeaways.
        - Be around 200 words.
        - Provide actionable insights for decision-makers.
        """,
        output_key="executive_summary",
    )
    print(f"  - AggregatorAgent created. Inputs: '{{tech_research}}', '{{health_research}}', '{{finance_research}}', Output: '{aggregator_agent.output_key}'")

    # The ParallelAgent runs all its sub-agents simultaneously.
    parallel_research_team = ParallelAgent(
        name="ParallelResearchTeam",
        sub_agents=[tech_researcher, health_researcher, finance_researcher],
    )
    print(f"  - ParallelAgent 'ParallelResearchTeam' created with sub-agents: {[a.name for a in parallel_research_team.sub_agents]}")

    # This SequentialAgent defines the high-level workflow:
    # 1. Run the parallel team first.
    # 2. Then run the aggregator.
    research_system = SequentialAgent(
        name="ResearchSystem",
        sub_agents=[parallel_research_team, aggregator_agent],
    )
    print(f"  - SequentialAgent 'ResearchSystem' created with sub-agents: {[a.name for a in research_system.sub_agents]}")
    return research_system


# --- Section 4: Loop Workflows (Story Refinement Loop) ---

def exit_loop_tool() -> dict:
    """
A simple Python function wrapped as a FunctionTool to signal loop exit.

This function is called by the RefinerAgent when the story critique is
"APPROVED", indicating that the iterative refinement process should stop.

Returns:
    dict: A status message indicating successful approval and loop exit.
"""
    return {"status": "approved", "message": "Story approved. Exiting refinement loop."}


def define_loop_story_refinement_system() -> SequentialAgent:
    """
    Defines a loop-based agent system for iterative story refinement.

    This system demonstrates how `LoopAgent` can be used for tasks requiring
    iterative improvement and feedback. It includes:
    - InitialWriterAgent: Creates the first draft.
    - CriticAgent: Reviews the story and provides feedback or approval.
    - RefinerAgent: Rewrites the story based on critique or calls `exit_loop_tool` if approved.
    - LoopAgent: Manages the iterative cycle of critique and refinement.

    Returns:
        SequentialAgent: The root sequential agent orchestrating initial writing and the refinement loop.
    """
    # This agent runs ONCE at the beginning to create the first draft.
    initial_writer_agent = Agent(
        name="InitialWriterAgent",
        model=GEMINI_MODEL,
        instruction="""Based on the user's prompt, write the first draft of a short story (around 100-150 words).

        Guidelines:
        - Create an engaging beginning, middle, and end.
        - Include at least one character and a clear plot.
        - Output only the story text, with no introduction or explanation.
        """,
        output_key="current_story",
    )
    print(f"  - InitialWriterAgent created. Output: '{initial_writer_agent.output_key}'")

    # This agent's only job is to provide feedback or the approval signal.
    critic_agent = Agent(
        name="CriticAgent",
        model=GEMINI_MODEL,
        instruction="""You are a constructive story critic. Review the story provided below.

        Story: {current_story}

        Evaluate the story's:
        - Plot coherence and engagement.
        - Character development.
        - Pacing and flow.
        - Overall quality.

        Response format:
        - If the story is well-written and complete, you MUST respond with the exact phrase: "APPROVED"
        - Otherwise, provide 2-3 specific, actionable suggestions for improvement.
        """,
        output_key="critique",
    )
    print(f"  - CriticAgent created. Input: '{{current_story}}', Output: '{critic_agent.output_key}'")

    # This agent refines the story based on critique OR calls the exit_loop_tool function.
    refiner_agent = Agent(
        name="RefinerAgent",
        model=GEMINI_MODEL,
        instruction="""You are a story refiner. You have a story draft and critique.

        Story Draft: {current_story}
        Critique: {critique}

        Your task is to analyze the critique:
        - IF the critique is EXACTLY "APPROVED", you MUST call the `exit_loop_tool` function and nothing else.
        - OTHERWISE, rewrite the story draft to fully incorporate the feedback from the critique.

        When rewriting:
        - Address all the critic's suggestions.
        - Maintain the original story's core elements.
        - Keep the word count similar (100-150 words).
        """,
        output_key="current_story",  # Overwrites the story with the new version
        tools=[FunctionTool(exit_loop_tool)],
    )
    print(f"  - RefinerAgent created with tool: '{refiner_agent.tools[0].name}'. Input: '{{current_story}}', '{{critique}}', Output: '{refiner_agent.output_key}'")

    # The LoopAgent contains the agents that will run repeatedly: Critic -> Refiner.
    story_refinement_loop = LoopAgent(
        name="StoryRefinementLoop",
        sub_agents=[critic_agent, refiner_agent],
        max_iterations=MAX_LOOP_ITERATIONS,  # Prevents infinite loops
    )
    print(f"  - LoopAgent 'StoryRefinementLoop' created with sub-agents: {[a.name for a in story_refinement_loop.sub_agents]}. Max iterations: {story_refinement_loop.max_iterations}")

    # The root agent is a SequentialAgent that defines the overall workflow: Initial Write -> Refinement Loop.
    story_pipeline = SequentialAgent(
        name="StoryPipeline",
        sub_agents=[initial_writer_agent, story_refinement_loop],
    )
    print(f"  - SequentialAgent 'StoryPipeline' created with sub-agents: {[a.name for a in story_pipeline.sub_agents]}")
    return story_pipeline


def _print_demonstration_divider():
    """Prints a clear visual divider between demonstrations."""
    print("\n" + "=" * 80 + "\n")


async def demo_multi_agent_system():
    """Demonstrates a multi-agent system with a coordinator."""
    print("--- (1/4) Demonstrating Multi-Agent Research System ---")
    multi_agent_system = define_multi_agent_research_system()
    multi_agent_runner = InMemoryRunner(agent=multi_agent_system)
    multi_agent_query = "What are the latest advancements managing PII data for a Snowflake Data Warehouse?"
    print(f"\n💬 User Query: '{multi_agent_query}'")
    print("🧠 Multi-Agent System is thinking...")
    multi_agent_response = await multi_agent_runner.run_debug(multi_agent_query)
    print("\n✅ Multi-Agent System execution complete.")
    print("--- Agent's Final Response ---")
    print(multi_agent_response)
    print("------------------------------")


async def demo_sequential_blog_pipeline():
    """Demonstrates a sequential agent pipeline."""
    print("--- (2/4) Demonstrating Sequential Blog Post Pipeline ---")
    blog_pipeline = define_sequential_blog_pipeline()
    blog_runner = InMemoryRunner(agent=blog_pipeline)
    blog_topic = "Write a blog post about the benefits of masking policies for a data warehouse dealing with PII data"
    print(f"\n📝 Blog Topic: '{blog_topic}'")
    print("🧠 Sequential Pipeline is processing...")
    blog_response = await blog_runner.run_debug(blog_topic)
    print("\n✅ Sequential Pipeline execution complete.")
    print("--- Final Blog Post ---")
    print(blog_response)
    print("-----------------------")


async def demo_parallel_research_system():
    """Demonstrates a parallel agent system for concurrent research."""
    print("--- (3/4) Demonstrating Parallel Research System ---")
    parallel_research_system = define_parallel_research_system()
    research_runner = InMemoryRunner(agent=parallel_research_system)
    research_task = "Run the daily executive briefing on Tech, Health, and Finance"
    print(f"\n📊 Research Task: '{research_task}'")
    print("🧠 Parallel Research System is running...")
    research_response = await research_runner.run_debug(research_task)
    print("\n✅ Parallel Research System execution complete.")
    print("--- Executive Summary ---")
    print(research_response)
    print("-------------------------")


async def demo_loop_story_refinement_system():
    """Demonstrates a loop-based agent system for iterative refinement."""
    print("--- (4/4) Demonstrating Loop-based Story Refinement System ---")
    story_pipeline = define_loop_story_refinement_system()
    story_runner = InMemoryRunner(agent=story_pipeline)
    story_prompt = "Write a short story about AI Engineers discovering a hidden code in ancient manuscripts"
    print(f"\n📖 Story Prompt: '{story_prompt}'")
    print("🧠 Story Refinement System is writing and refining...")
    story_response = await story_runner.run_debug(story_prompt)
    print("\n✅ Story Refinement System execution complete.")
    print("--- Final Story ---")
    print(story_response)
    print("-------------------")


async def main() -> None:
    """
    The main asynchronous entry point for demonstrating various ADK agent architectures.

    This function orchestrates the setup of the environment and then sequentially
    runs demonstrations of multi-agent, sequential, parallel, and loop-based
    agent systems, pausing between each one for clarity.
    """
    print("--- Starting Advanced Agent Architectures Demonstrations ---")
    setup_environment()

    await demo_multi_agent_system()
    _print_demonstration_divider()

    await demo_sequential_blog_pipeline()
    _print_demonstration_divider()

    await demo_parallel_research_system()
    _print_demonstration_divider()

    await demo_loop_story_refinement_system()

    print("\n--- All Advanced Agent Architectures Demonstrations Complete ---")


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
