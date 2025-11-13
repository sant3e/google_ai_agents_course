#!/usr/bin/env python3
"""
Comprehensive test script to verify that all refactored scripts can properly import
the initial_setup module and access its exports.
"""

import sys
import importlib.util
import re

def test_script_imports(script_name):
    """Test if a script can be imported and has access to initial_setup exports."""
    print(f"\n--- Testing {script_name} ---")
    
    results = {
        "correct_import": False,
        "both_items_imported": False,
        "gemini_model_used": False,
        "setup_environment_used": False,
        "executes_without_error": False
    }
    
    try:
        # Read the script content
        with open(script_name, 'r') as f:
            content = f.read()
        
        # Check for the correct import statement format
        # For Day1 scripts, we expect GEMINI_MODEL and setup_environment
        # For Day2 scripts, we expect GEMINI_MODEL, setup_environment, asyncio, and sys
        if script_name.startswith("Day1"):
            expected_import = "from initial_setup import GEMINI_MODEL, setup_environment"
            if expected_import in content:
                print("✅ Correct import statement found: from initial_setup import GEMINI_MODEL, setup_environment")
                results["correct_import"] = True
            elif "from training_setup import" in content:
                print("❌ Still using old training_setup import")
            elif "from initial_setup import" in content:
                print("⚠️  Found initial_setup import but not the expected format")
                
                # Check if both items are imported but in a different format
                if "GEMINI_MODEL" in content and "setup_environment" in content:
                    print("✅ Both GEMINI_MODEL and setup_environment are imported (different format)")
                    results["both_items_imported"] = True
            else:
                print("❌ No initial_setup import found")
        elif script_name.startswith("Day2"):
            expected_import = "from initial_setup import GEMINI_MODEL, setup_environment, asyncio, sys"
            if expected_import in content:
                print("✅ Correct import statement found: from initial_setup import GEMINI_MODEL, setup_environment, asyncio, sys")
                results["correct_import"] = True
            elif "from training_setup import" in content:
                print("❌ Still using old training_setup import")
            elif "from initial_setup import" in content:
                print("⚠️  Found initial_setup import but not the expected format")
                
                # Check if all items are imported but in a different format
                if ("GEMINI_MODEL" in content and "setup_environment" in content and
                    "asyncio" in content and "sys" in content):
                    print("✅ All required items (GEMINI_MODEL, setup_environment, asyncio, sys) are imported (different format)")
                    results["both_items_imported"] = True
            else:
                print("❌ No initial_setup import found")
            
        # Check if required items are imported
        if "from initial_setup import" in content:
            if script_name.startswith("Day1"):
                if "GEMINI_MODEL" in content and "setup_environment" in content:
                    print("✅ Both GEMINI_MODEL and setup_environment are imported")
                    results["both_items_imported"] = True
                else:
                    print("❌ Missing either GEMINI_MODEL or setup_environment in import")
            elif script_name.startswith("Day2"):
                if ("GEMINI_MODEL" in content and "setup_environment" in content and
                    "asyncio" in content and "sys" in content):
                    print("✅ All required items (GEMINI_MODEL, setup_environment, asyncio, sys) are imported")
                    results["both_items_imported"] = True
                else:
                    print("❌ Missing required items in import (need GEMINI_MODEL, setup_environment, asyncio, sys)")
                
            # Check if they are actually used in the code
            # Remove the import line to avoid false positives
            content_without_imports = re.sub(r'from initial_setup import.*', '', content)
            
            if "GEMINI_MODEL" in content_without_imports:
                print("✅ GEMINI_MODEL is used in the code")
                results["gemini_model_used"] = True
            else:
                print("❌ GEMINI_MODEL is imported but not used")
                
            if "setup_environment()" in content:
                print("✅ setup_environment() is called in the code")
                results["setup_environment_used"] = True
            else:
                print("❌ setup_environment() is imported but not called")
        
        # Try to actually execute the module's imports
        try:
            spec = importlib.util.spec_from_file_location(script_name, script_name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print("✅ Script imports successfully without errors")
            results["executes_without_error"] = True
        except Exception as e:
            print(f"❌ Error importing or executing {script_name}: {e}")
            
        # Return True only if all checks pass
        return all(results.values())
        
    except Exception as e:
        print(f"❌ Error reading {script_name}: {e}")
        return False

def main():
    """Test all four refactored scripts."""
    scripts = ["Day1a.py", "Day1b.py", "Day2a.py", "Day2b.py"]
    results = {}
    detailed_results = {}
    
    print("Testing refactored scripts for initial_setup module integration...")
    
    for script in scripts:
        passed = test_script_imports(script)
        results[script] = passed
    
    print("\n=== SUMMARY ===")
    all_passed = True
    for script, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{script}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All scripts passed the import tests!")
    else:
        print("\n⚠️ Some scripts have issues that need to be fixed.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)