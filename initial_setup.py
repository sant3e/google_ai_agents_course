# -*- coding: utf-8 -*-
"""
Shared setup module for ADK training examples.

This module contains common imports, constants, and setup functions
used across all training day examples to avoid code duplication.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# --- Constants and Configuration ---

GEMINI_MODEL = "gemini-2.5-flash-lite"
"""Specifies the Gemini model to be used for the agent's reasoning."""


def setup_environment() -> None:
    """
    Loads environment variables and configures the Google API key.

    This function is a critical first step that ensures the application has
    the necessary credentials to interact with Google's services. It loads
    variables from a `.env` file and sets the required environment variables
    for the ADK and GenAI libraries.

    Raises:
        ValueError: If the GOOGLE_API_KEY is not found in the environment.
    """
    load_dotenv()
    google_api_key = os.environ.get("GOOGLE_API_KEY")

    if not google_api_key:
        raise ValueError(
            "Missing GOOGLE_API_KEY. Ensure it's set in your .env file."
        )

    # The ADK and underlying libraries use these environment variables for auth.
    os.environ["GOOGLE_API_KEY"] = google_api_key
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("✅ Environment configured successfully.")