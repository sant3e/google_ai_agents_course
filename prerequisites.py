# -*- coding: utf-8 -*-
"""
Prerequisites: Package Installation Helper.

This script ensures all required packages are installed for the Google ADK examples.
It's designed to be imported by other scripts to handle package installation needs.

Usage:
    import prerequisites  # This will install any missing packages

Prerequisites:
- python-dotenv
- google-adk
- google-genai
- mcp
"""

import subprocess
import sys


def install_required_packages():
    """
    Installs required packages if they are not already installed.

    This function checks for the presence of required packages and installs
    them if they're missing, allowing other scripts to run without manual
    package installation.
    """
    required_packages = ["python-dotenv", "google-adk", "google-genai", "mcp", "pandas"]
    a2a_package = "google-adk[a2a]"  # Special package with A2A dependencies

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} is already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", package])
            print(f"✅ {package} installed successfully")

    # Install A2A dependencies separately (with extras)
    try:
        # Try to import the core ADK components first
        import google.adk.agents.remote_a2a_agent
        print("✅ A2A dependencies are already installed")
    except (ImportError, ModuleNotFoundError):
        print(f"Installing {a2a_package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", a2a_package])
        print(f"✅ {a2a_package} installed successfully")


if __name__ == "__main__":
    install_required_packages()