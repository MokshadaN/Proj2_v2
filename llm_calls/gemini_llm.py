# llm_calls/gemini_llm.py
import os
import json
from google import genai
from google.genai.types import GenerateContentConfig
import re

GEMINI_MODEL_DEFAULT = "gemini-2.5-flash"

def call_gemini_llm(system_prompt: str, user_prompt: str, content=None) -> str:
    print("call  to gemini api")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)

    # Combine user prompt and content into a single string for the 'contents' parameter
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    if content:
        content_str = json.dumps(content, indent=2) if isinstance(content, (dict, list)) else str(content)
        full_prompt += f"\n\n--- Additional Context ---\n{content_str}"

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_DEFAULT,
            contents=full_prompt,
            config=GenerateContentConfig(temperature=0)
        )

        if not response.candidates or not response.candidates[0].content.parts:
            raise ValueError("No response from Gemini model.")
        print("got the response")
        return response.candidates[0].content.parts[0].text.strip()

    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {e}")


def clean_llm_code(code: str) -> str:
    """
    Remove markdown fences from LLM output without breaking syntax.
    """
    # Remove triple backticks and optional language hint
    code = re.sub(r"^```[a-zA-Z0-9]*\n?", "", code.strip(), flags=re.MULTILINE)
    code = re.sub(r"\n?```$", "", code.strip(), flags=re.MULTILINE)
    return code.strip()

# Example usage (optional, for testing this file directly)
if __name__ == '__main__':
    # This requires the GOOGLE_API_KEY to be set in your environment
    try:
        
        system = "You are a Python code generator who only responds with code."
        user = "Write a script to create a pandas DataFrame with two columns, 'A' and 'B', and print its head."
        
        generated_code = call_gemini_llm(system, user)
        print("--- Raw LLM Output ---")
        print(generated_code)
        
        cleaned_code = clean_llm_code(generated_code)
        print("\n--- Cleaned Code ---")
        print(cleaned_code)

    except ImportError:
        print("Could not import clean_llm_code for testing.")
    except (ValueError, RuntimeError) as e:
        print(e)
