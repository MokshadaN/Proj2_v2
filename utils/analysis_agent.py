from utils.anthropic_analysis import SimpleAnthropicAnalysisAgent  # Or just put the full class here
import os 
class AnalysisAgent:
    def __init__(self, api_key: str = None):
        # If no api_key passed, load from env var or config
        from os import getenv
        if not api_key:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")
        self.agent = SimpleAnthropicAnalysisAgent(api_key=api_key)
    
    def run_analysis(self, analysis_plan, data_handle):
        return self.agent.run_analysis(analysis_plan, data_handle)
