#!/usr/bin/env python3
"""
Plan Execution Script - Full Plan Mode

This version passes the entire plan.json to Anthropic LLM,
receives a single script, executes it in Docker, and returns results.
"""

import json
import os
import sys
import logging
import time
from typing import Any, Dict, List
from pathlib import Path
import traceback

# Import custom modules
from llm_calls.anthro_llm import AnthropicLLM, clean_llm_code
from llm_calls.docker_execute import DockerScriptRunner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("plan_execution.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PlanExecutionError(Exception):
    pass

class PlanExecutor:
    def __init__(self, plan_path: str, working_dir: str = None):
        self.plan_path = Path(plan_path)
        self.working_dir = Path(working_dir) if working_dir else self.plan_path.parent
        self.llm_client = AnthropicLLM()
        self.docker_runner = DockerScriptRunner()
        self.plan_data = None
        self.execution_history = []

        logger.info(f"PlanExecutor initialized with plan: {self.plan_path}")
        logger.info(f"Working directory: {self.working_dir}")

    def load_and_validate_plan(self) -> Dict[str, Any]:
        try:
            with open(self.plan_path, 'r', encoding='utf-8') as f:
                plan_data = json.load(f)

            required_keys = ['sourcing_plan', 'analysis_plan', 'final_output_format']
            for key in required_keys:
                if key not in plan_data:
                    raise PlanExecutionError(f"Missing key in plan: {key}")

            logger.info("Plan loaded and validated successfully")
            return plan_data

        except Exception as e:
            raise PlanExecutionError(f"Failed to load plan: {e}")

    def create_system_prompt(self) -> str:
        return (
            "You are an expert Python data engineer and data analyst. "
            "Your task is to generate a complete Python script that implements the full execution plan provided. "
            "Do NOT include explanations or markdown. Output only the executable Python code.\n\n"
            "REQUIREMENTS:\n"
            "- Implement both data sourcing and analysis as defined\n"
            "- Use robust error handling and logging\n"
            "- Ensure proper validation (e.g., empty dataframes, column types)\n"
            "- Output final analysis results clearly (one per line or JSON object)\n"
            "- Save images as base64 if required\n"
        )

    def create_user_prompt(self, plan_data: Dict[str, Any]) -> str:
        return (
            "Below is a plan for data sourcing and analysis. "
            "Generate a full executable Python script that completes all defined tasks.\n\n"
            "Plan:\n"
            + json.dumps(plan_data, indent=2)
        )

    def execute_whole_plan(self) -> List[str]:
        logger.info("Passing full plan to Anthropic LLM...")

        system_prompt = self.create_system_prompt()
        user_prompt = self.create_user_prompt(self.plan_data)

        generated_code = self.llm_client.call_anthropic_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            content=self.plan_data
        )

        cleaned_code = clean_llm_code(generated_code)
        if not cleaned_code.strip():
            raise PlanExecutionError("LLM returned empty code.")

        logger.info("Executing generated script in Docker...")
        result = self.docker_runner.call_python_script(
            script=cleaned_code,
            working_dir=str(self.working_dir),
            timeout_sec=300
        )

        if result["success"]:
            stdout = result.get("stdout", "").strip()
            return stdout.splitlines()
        else:
            stderr = result.get("stderr", result.get("error", "Unknown error"))
            raise PlanExecutionError(f"Execution failed: {stderr}")

    def execute_plan(self) -> List[str]:
        try:
            logger.info("Loading plan...")
            self.plan_data = self.load_and_validate_plan()
            results = self.execute_whole_plan()

            format_type = self.plan_data['final_output_format'].get('format')
            if format_type == 'array_of_strings':
                results = [str(r) for r in results]

            logger.info("Plan execution completed.")
            return results

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            logger.error(traceback.format_exc())
            raise PlanExecutionError(str(e))

    def generate_execution_report(self) -> str:
        return (
            "=== EXECUTION REPORT ===\n"
            f"Plan: {self.plan_path}\n"
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Working Directory: {self.working_dir}\n"
            f"History Entries: {len(self.execution_history)}\n"
        )

def main():
    if len(sys.argv) < 2:
        print("Usage: python execute_plan.py <plan.json> [working_directory]")
        sys.exit(1)

    plan_path = sys.argv[1]
    working_dir = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        executor = PlanExecutor(plan_path, working_dir)
        results = executor.execute_plan()

        print("\n" + "="*60)
        print("EXECUTION COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Results ({len(results)} items):")
        for i, res in enumerate(results, 1):
            print(f"{i}. {res}")

        # Save results
        results_file = executor.working_dir / "execution_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'execution_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'plan_file': str(plan_path),
                'results': results,
                'execution_history': executor.execution_history
            }, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {results_file}")

        report_file = executor.working_dir / "execution_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(executor.generate_execution_report())

        print(f"Execution report saved to: {report_file}")

    except PlanExecutionError:
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
