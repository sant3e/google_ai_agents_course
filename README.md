# 5-Day AI Agents Intensive Course with Google

This repository contains my personal implementations and notes from the 5-Day AI Agents Intensive Course developed by Google's ML researchers and engineers, restructured with a modular, functional programming approach for better code organization and reusability.

## Course Overview

The 5-Day AI Agents Intensive is an online course designed to help developers explore the foundations and practical applications of AI agents. The course covers:

- Core components of AI agents: models, tools, orchestration, memory, and evaluation
- How agents move beyond LLM prototypes to become production-ready systems
- Hands-on examples, codelabs, and practical implementations

## Resources

- [Agent Development Kit](https://google.github.io/adk-docs/) - Official documentation for Google's Agent Development Kit

## Repository Content

This repository follows the course structure with both the original Jupyter notebooks and my own Python scripts for each day:

- **Day 1**: Introduction to AI Agents fundamentals
  - `Day1a.ipynb` / `Day1a.py` - First part of Day 1 exercises (Basic "Hello, World!" Agent with Google's ADK)
  - `Day1b.ipynb` / `Day1b.py` - Second part of Day 1 exercises (Advanced Agent Architectures with Google's ADK)

- **Day 2**: Building Agents with Tools
  - `Day2a.ipynb` / `Day2a.py` - First part of Day 2 exercises (Building Agents with Custom Tools)
  - `Day2b.ipynb` / `Day2b.py` - Second part of Day 2 exercises (Advanced Agent Tool Patterns)

## Additional Python Files

The repository also includes the following utility Python scripts:

- **prerequisites.py**: Script to check and install required dependencies for the course. Run this once, first.
- **initial_setup.py**: Shared setup module containing common imports, GEMINI_MODEL constant, and setup_environment function. No need to run this, it's just a module used by lessons.
- **test_imports.py**: Comprehensive test script to verify all imports work correctly. Run when needed to test imports.



## Project Approach

This repository represents a personal adaptation of the original course material with a focus on:

- **Modularization**: Code is organized into reusable components and utilities, making it easier to maintain and extend
- **Functional Programming**: Functions are designed to be pure and composable, reducing side effects and improving testability
- **Clear Separation of Concerns**: Each module has a specific responsibility, with shared functionality extracted into utility modules
- **Enhanced Reusability**: Common patterns and setup code are centralized to avoid duplication across different exercises

## Learning Goals

By the end of this course, you'll be able to:
- Build, evaluate, and deploy AI agents that solve real-world problems
- Understand the architecture and components of production-ready AI agents
- Implement practical AI agent solutions using modern tools and frameworks
- Apply modular design principles to AI agent development for better maintainability and scalability

## NOTE:
- To simulate the interaction with the script in the same way you do with a Jupyter Notebook, you can use Interactive Window in VSC.
- select blocks of code and SHIFT+ENTER to trigger the Interactive Window.
- Usually i select the entire body of one function and run it, then move to next one