import json
from google import genai
from google.genai import types
from typing import Any
import os
from utils.prompts import PromptManager

class PlannerAgent:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        self.client = genai.Client(api_key=api_key)
        self.prompt_manager = PromptManager()

    def create_plan(self, questions_content: str, uploaded_files_str: str) -> dict[str, Any]:
        system_prompt, user_prompt = self.prompt_manager.planner_agent_prompt(questions_content, uploaded_files_str)

        if not user_prompt:
            user_prompt = f"""
            Here is the user's task from questions.txt:
            ---
            {questions_content}
            ---

            And here are the files available for analysis:
            ---
            {uploaded_files_str}
            ---
            """

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json"
                ),
                contents=user_prompt
            )
            plan_json_string = response.text
            return json.loads(plan_json_string)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from Gemini: {e}")
            print(f"LLM output was: {response.text}")
            raise ValueError("Could not generate a valid JSON plan from the LLM.")
        except Exception as e:
            print(f"Gemini planning failed: {e}")
            raise ValueError("Could not generate a valid plan from the LLM.")

# Example usage (optional, for testing this file directly)
if __name__ == '__main__':
    # This requires the ANTHROPIC_API_KEY to be set in your environment
    try:
        planner = PlannerAgent()
        
        mock_questions = "Scrape https://example.com, find the main table, and calculate the average of the 'value' column."
        mock_files = "data.csv"
        
        plan = planner.create_plan(mock_questions, mock_files)
        
        print("\n--- Generated Plan ---")
        print(json.dumps(plan, indent=2))

    except (ValueError) as e:
        print(e)
