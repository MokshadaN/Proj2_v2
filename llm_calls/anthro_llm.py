import os
import re
import json
import anthropic

class AnthropicLLM:
    """
    A client for interacting with the Anthropic (Claude) API.
    """
    def __init__(self, api_key: str = None):
        """
        Initializes the Anthropic client.
        
        Args:
            api_key (str, optional): The Anthropic API key. If not provided, it will
                                     be read from the ANTHROPIC_API_KEY environment variable.
        """
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def call_anthropic_llm(self, system_prompt: str, user_prompt: str, content=None, 
                           model: str = "claude-3-haiku-20240307", max_tokens=4096) -> str:
        """
        Calls the Anthropic API with a system prompt, user prompt, and optional content.

        Args:
            system_prompt (str): The system-level instructions for the model.
            user_prompt (str): The user's specific request.
            content (any, optional): Additional content (like schema or data samples) to be sent.
            model (str): The model to use (e.g., claude-3-sonnet-20240229).
            max_tokens (int): The maximum number of tokens to generate.

        Returns:
            str: The text content of the LLM's response.
        """
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        if content:
            # Add the structured content as a separate message part for clarity
            messages.append({
                "role": "user", 
                "content": json.dumps(content, indent=2) if isinstance(content, (dict, list)) else str(content)
            })

        try:
            response = self.client.messages.create(
                model=model,
                system=system_prompt,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1, # Low temperature for deterministic code generation
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            raise

def clean_llm_code(raw_code: str) -> str:
    """
    Cleans LLM-generated code by removing markdown fences and non-code commentary.
    
    Args:
        raw_code (str): The raw string output from the LLM.
        
    Returns:
        str: The cleaned, executable Python code.
    """
    if not isinstance(raw_code, str):
        return ""

    # Remove markdown fences (e.g., ```python ... ```)
    code = re.sub(r"^```(?:python)?", "", raw_code.strip(), flags=re.MULTILINE)
    code = re.sub(r"```$", "", code.strip(), flags=re.MULTILINE)
    
    # Remove any leading text before the first import or variable assignment.
    # This handles cases where the LLM says "Here is the code:"
    match = re.search(r"(^import\s|^from\s|^\w+\s*=|^\s*def\s)", code, re.MULTILINE)
    if match:
        code = code[match.start():]
    
    return code.strip()

# Example usage (optional, for testing this file directly)
if __name__ == '__main__':
    # This requires the ANTHROPIC_API_KEY to be set in your environment
    llm_client = AnthropicLLM()
    
    system = "You are a Python code generator."
    user = "Write a script to print 'hello world'."
    
    generated_code = llm_client.call_anthropic_llm(system, user)
    print("--- Raw LLM Output ---")
    print(generated_code)
    
    cleaned_code = clean_llm_code(generated_code)
    print("\n--- Cleaned Code ---")
    print(cleaned_code)
