import os
import json
import tempfile
import pandas as pd

# Import the components we want to test
from llm_calls.docker_execute import DockerScriptRunner
# Change the import to use the Gemini LLM function
from llm_calls.gemini_llm import call_gemini_llm

def create_dummy_csv(directory: str) -> str:
    """Creates a simple CSV file for testing and returns its path."""
    filepath = os.path.join(directory, "test_data.csv")
    df = pd.DataFrame({
        "product_id": ["A", "B", "C", "D"],
        "sales": [100, 150, 80, 220]
    })
    df.to_csv(filepath, index=False)
    print(f"Created dummy CSV at: {filepath}")
    return filepath

def main():
    """
    Runs an end-to-end test of the LLM code generation and Docker execution pipeline.
    """
    # --- 1. Setup the Test Environment ---
    # Create a temporary directory to act as our 'uploads' folder
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Create a dummy data file inside the temporary directory
        csv_path = create_dummy_csv(temp_dir)
        
        # --- 2. Prepare the LLM Prompt ---
        # This simulates the context the AnalysisAgent would create
        test_context = {
            "task": {
                "instruction": "Calculate the average sales from the provided dataset."
            },
            "data_path": os.path.basename(csv_path), # The script will use this relative path
            "schema": {"product_id": "object", "sales": "int64"},
            "data_sample": [{"product_id": "A", "sales": 100}, {"product_id": "B", "sales": 150}]
        }
        
        # Define the prompts directly in the test script
        system_prompt = """
You are an expert Python data scientist. You will be given a task and context about a dataset.
Your job is to write a self-contained, executable Python script that performs the requested analysis and prints a single JSON object to standard output.

**CRITICAL REQUIREMENTS**:
- The script MUST be completely self-contained and import all necessary libraries (pandas, json).
- The script MUST read the data from the file path provided in the context (e.g., `pd.read_csv('test_data.csv')`).
- The final output of the script must be a single JSON object printed to stdout (e.g., `print(json.dumps(result))`).
- Do not use any variables that are not defined within the script itself.
"""

        user_prompt = f"""
Write a Python script to perform the following task.

--- TASK CONTEXT ---
{json.dumps(test_context, indent=2)}
---

Write the complete, self-contained Python script now.
"""
        
        # --- 3. Generate Code using the LLM ---
        print("\n--- Step 1: Generating Python code via Gemini LLM ---")
        try:
            # Call the Gemini function directly
            generated_script = call_gemini_llm(system_prompt, user_prompt)
            print("Code generated successfully:")
            print("-" * 30)
            print(generated_script)
            print("-" * 30)
        except Exception as e:
            print(f"Failed to generate code: {e}")
            return

        # --- 4. Execute the Generated Code in Docker ---
        print("\n--- Step 2: Executing generated code in Docker ---")
        try:
            # Initialize the runner (it will use the 'data-analyst-agent:latest' image)
            docker_runner = DockerScriptRunner()
            
            # The working_dir is the absolute path to our temporary folder
            # This directory will be mounted into the container
            result = docker_runner.call_python_script(
                script=generated_script,
                working_dir=temp_dir 
            )

            # --- 5. Display the Results ---
            print("\n--- Step 3: Analyzing Execution Result ---")
            if result.get("success"):
                print("✅ Execution Successful!")
                print("\n--- STDOUT from Container ---")
                # Try to parse the stdout as JSON, as the prompt requested
                try:
                    stdout_json = json.loads(result['stdout'])
                    print(json.dumps(stdout_json, indent=2))
                except json.JSONDecodeError:
                    print("Warning: STDOUT was not valid JSON.")
                    print(result['stdout'])
                
            else:
                print("❌ Execution Failed!")
                print("\n--- STDERR from Container ---")
                print(result.get("stderr") or result.get("error", "No error message captured."))

        except Exception as e:
            print(f"An error occurred during Docker execution: {e}")
            print("Please ensure your Docker image 'data-analyst-agent:latest' is built and the Docker daemon is running.")

if __name__ == "__main__":
    # This requires GOOGLE_API_KEY to be set in your .env file
    # and your Docker image to be built.
    main()
