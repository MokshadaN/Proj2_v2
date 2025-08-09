#!/usr/bin/env python3
"""
Plan Execution Script with Anthropic LLM Integration and Feedback Loop

This script executes a plan.json file by:
1. Loading and validating the plan structure
2. Generating Python code using Anthropic LLM for data sourcing and analysis
3. Executing code in Docker with error handling and retry mechanisms
4. Implementing comprehensive validation and precautions
5. Providing detailed feedback loop for iterative improvement
"""

import json
import os
import sys
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import traceback

# Import existing modules
from llm_calls.anthro_llm import AnthropicLLM, clean_llm_code
from llm_calls.docker_execute import DockerScriptRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('plan_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PlanExecutionError(Exception):
    """Custom exception for plan execution errors"""
    pass

class PlanExecutor:
    """
    Comprehensive plan executor with LLM integration, Docker execution,
    validation, and feedback loop mechanisms.
    """
    
    def __init__(self, plan_path: str, working_dir: str = None, max_retries: int = 3):
        """
        Initialize the plan executor.
        
        Args:
            plan_path: Path to the plan.json file
            working_dir: Working directory for file operations
            max_retries: Maximum number of retries for failed operations
        """
        self.plan_path = Path(plan_path)
        self.working_dir = Path(working_dir) if working_dir else self.plan_path.parent
        self.max_retries = max_retries
        
        # Initialize components
        self.llm_client = AnthropicLLM()
        self.docker_runner = DockerScriptRunner()
        
        # Execution state
        self.plan_data = None
        self.execution_history = []
        self.results = {}
        
        logger.info(f"PlanExecutor initialized with plan: {self.plan_path}")
        logger.info(f"Working directory: {self.working_dir}")

    def load_and_validate_plan(self) -> Dict[str, Any]:
        """
        Load and validate the plan.json file structure.
        
        Returns:
            Validated plan data
            
        Raises:
            PlanExecutionError: If plan is invalid or missing required fields
        """
        try:
            with open(self.plan_path, 'r', encoding='utf-8') as f:
                plan_data = json.load(f)
            
            # Validate required top-level keys
            required_keys = ['sourcing_plan', 'analysis_plan', 'final_output_format']
            for key in required_keys:
                if key not in plan_data:
                    raise PlanExecutionError(f"Missing required key in plan: {key}")
        
            logger.info("Plan validation completed successfully")
            return plan_data
            
        except json.JSONDecodeError as e:
            raise PlanExecutionError(f"Invalid JSON in plan file: {e}")
        except FileNotFoundError:
            raise PlanExecutionError(f"Plan file not found: {self.plan_path}")
        except Exception as e:
            raise PlanExecutionError(f"Error loading plan: {e}")

    def create_system_prompt(self, stage: str, context: Dict[str, Any] = None) -> str:
        """
        Create comprehensive system prompts for different execution stages.
        
        Args:
            stage: Execution stage ('sourcing', 'analysis', 'retry')
            context: Additional context for prompt generation
            
        Returns:
            Formatted system prompt
        """
        base_prompt = """You are an expert Python data analyst and programmer. Your task is to generate clean, efficient, and robust Python code that follows best practices.

CRITICAL REQUIREMENTS:
1. Generate ONLY executable Python code - no explanations, comments, or markdown
2. Include comprehensive error handling and validation
3. Use appropriate libraries as specified in the plan
4. Implement proper data type checking and conversion
5. Handle missing values and edge cases gracefully
6. Generate code that is production-ready and defensive

VALIDATION REQUIREMENTS:
- Always check if scraped/loaded data is not empty before proceeding
- Validate data types match expected schema
- Handle network errors, file not found errors, and parsing errors
- Use try-catch blocks around critical operations
- Print informative error messages for debugging

LIBRARIES AND IMPORTS:
- Import all required libraries at the top
- Use absolute imports when possible
- Handle import errors gracefully"""

        if stage == "sourcing":
            return base_prompt + """

DATA SOURCING SPECIFIC REQUIREMENTS:
- For web scraping: validate that the response is successful and contains expected content
- For CSV files: check file exists and has expected columns
- After data loading: validate the DataFrame is not empty and has expected shape
- Implement data type conversion as specified in the schema
- Apply error handling strategies as specified in the plan
- Save intermediate results for debugging if needed
- Print data summary (shape, columns, sample rows) for verification"""

        elif stage == "analysis":
            return base_prompt + """

DATA ANALYSIS SPECIFIC REQUIREMENTS:
- Assume data variables from sourcing stage are available in global scope
- Validate input data exists and is not empty before analysis
- Handle different output formats (number, string, json_object, base64_image)
- For visualizations: save plots as high-quality images and encode as base64
- For numerical results: ensure proper formatting and precision
- For text results: provide clear, concise answers
- Include error handling for mathematical operations (division by zero, etc.)"""

        elif stage == "retry":
            error_context = context.get('error_info', 'Unknown error') if context else 'Unknown error'
            prev_code = context.get('previous_code', '') if context else ''
            
            return base_prompt + f"""

ERROR RECOVERY AND RETRY REQUIREMENTS:
- Previous attempt failed with error: {error_context}
- Previous code that failed: {prev_code}
- Analyze the error and fix the root cause
- Add additional validation and error checking
- Use alternative approaches if the original method fails
- Implement more robust error handling
- Add debugging print statements to help identify issues"""

        return base_prompt

    def create_user_prompt(self, task_type: str, task_data: Dict[str, Any], 
                          context: Dict[str, Any] = None) -> str:
        """
        Create specific user prompts for different task types.
        
        Args:
            task_type: Type of task ('sourcing' or 'analysis')
            task_data: Task-specific data from plan
            context: Additional context (e.g., previous results, errors)
            
        Returns:
            Formatted user prompt
        """
        if task_type == "sourcing":
            prompt = f"""Generate Python code to {task_data['description']}

DATA SOURCE:
- URL: {task_data['data_source']['url']}
- Type: {task_data['data_source']['type']}
- Source Type: {task_data['source_type']}

EXTRACTION PARAMETERS:
{json.dumps(task_data.get('extraction_params', {}), indent=2)}

EXPECTED OUTPUT:
- Variable Name: {task_data['output']['variable_name']}
- Expected Format: {task_data.get('expected_format', 'pandas_dataframe')}
- Schema: {json.dumps(task_data['output']['schema'], indent=2)}

ERROR HANDLING:
{json.dumps(task_data.get('error_handling', {}), indent=2)}

VALIDATION RULES:
{json.dumps(task_data.get('validation_rules', []), indent=2)}

REQUIRED LIBRARIES:
{', '.join(task_data.get('libraries', []))}

Generate complete Python code that implements this data sourcing task with robust error handling and validation."""

        elif task_type == "analysis":
            available_vars = context.get('available_variables', []) if context else []
            prompt = f"""Generate Python code for the following analysis task:

TASK: {task_data['instruction']}
OUTPUT FORMAT: {task_data['output_format']}

AVAILABLE DATA VARIABLES:
{', '.join(available_vars) if available_vars else 'Check plan for variable names'}

REQUIREMENTS:
- Use the available data variables from previous sourcing steps
- Generate output in the exact format specified: {task_data['output_format']}
- Handle edge cases and potential errors
- For base64_image format: save plot and convert to base64 string
- For json_object format: return valid JSON string
- For number format: return clean numeric value
- For string format: return clear, concise text answer"""

        else:
            prompt = f"Generate Python code for: {task_data}"

        # Add retry context if available
        if context and context.get('retry_attempt'):
            error_info = context.get('error_info', 'Unknown error')
            prev_output = context.get('previous_output', 'No previous output')
            prompt += f"""

THIS IS A RETRY ATTEMPT (#{context['retry_attempt']})
Previous attempt failed with:
ERROR: {error_info}
PREVIOUS OUTPUT: {prev_output}

Please fix the issues and provide a more robust solution."""

        return prompt

    def execute_sourcing_plan(self) -> Dict[str, Any]:
        """
        Execute the data sourcing portion of the plan.
        
        Returns:
            Dictionary containing sourcing results and available variables
        """
        logger.info("Starting data sourcing phase...")
        sourcing_results = {}
        available_variables = []
        
        for i, source_config in enumerate(self.plan_data['sourcing_plan']):
            logger.info(f"Processing data source {i+1}/{len(self.plan_data['sourcing_plan'])}")
            logger.info(f"Description: {source_config['description']}")
            
            var_name = source_config['output']['variable_name']
            
            # Execute with retry logic
            success, result = self._execute_with_retry(
                task_type="sourcing",
                task_data=source_config,
                task_id=f"source_{i}"
            )
            
            if success:
                sourcing_results[var_name] = result
                available_variables.append(var_name)
                logger.info(f"✓ Successfully processed data source: {var_name}")
            else:
                logger.error(f"✗ Failed to process data source {i}: {result}")
                raise PlanExecutionError(f"Critical failure in data sourcing: {result}")
        
        logger.info(f"Data sourcing completed. Available variables: {available_variables}")
        return {
            'results': sourcing_results,
            'available_variables': available_variables
        }

    def execute_analysis_plan(self, available_variables: List[str]) -> List[Any]:
        """
        Execute the analysis portion of the plan.
        
        Args:
            available_variables: List of variable names available from sourcing
            
        Returns:
            List of analysis results in the specified output format
        """
        logger.info("Starting data analysis phase...")
        analysis_results = []
        
        for task in self.plan_data['analysis_plan']:
            logger.info(f"Processing analysis task {task['task_id']}: {task['qid']}")
            logger.info(f"Instruction: {task['instruction']}")
            
            context = {'available_variables': available_variables}
            
            # Execute with retry logic
            success, result = self._execute_with_retry(
                task_type="analysis",
                task_data=task,
                task_id=task['qid'],
                context=context
            )
            
            if success:
                analysis_results.append(result.get('final_result', result))
                logger.info(f"✓ Completed analysis task: {task['qid']}")
            else:
                logger.error(f"✗ Failed analysis task {task['qid']}: {result}")
                # For analysis tasks, we might continue with placeholder results
                analysis_results.append(f"ERROR: {result}")
        
        logger.info("Data analysis phase completed")
        return analysis_results

    def _execute_with_retry(self, task_type: str, task_data: Dict[str, Any], 
                           task_id: str, context: Dict[str, Any] = None) -> Tuple[bool, Any]:
        """
        Execute a task with retry logic and feedback loop.
        
        Args:
            task_type: Type of task ('sourcing' or 'analysis')
            task_data: Task configuration data
            task_id: Unique identifier for the task
            context: Additional context for execution
            
        Returns:
            Tuple of (success: bool, result: Any)
        """
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"Executing {task_id} (attempt {attempt + 1}/{self.max_retries + 1})")
                
                # Create prompts
                retry_context = None
                if attempt > 0:
                    # Add retry context with previous error information
                    retry_context = {
                        'retry_attempt': attempt,
                        'error_info': getattr(self, '_last_error', 'Unknown error'),
                        'previous_code': getattr(self, '_last_code', ''),
                        'previous_output': getattr(self, '_last_output', '')
                    }
                    if context:
                        retry_context.update(context)
                
                system_prompt = self.create_system_prompt(
                    'retry' if attempt > 0 else task_type,
                    retry_context
                )
                user_prompt = self.create_user_prompt(
                    task_type, 
                    task_data, 
                    retry_context or context
                )
                
                # Generate code using LLM
                logger.info("Generating code using Anthropic LLM...")
                generated_code = self.llm_client.call_anthropic_llm(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    content=task_data
                )
                
                # Clean and validate generated code
                cleaned_code = clean_llm_code(generated_code)
                if not cleaned_code.strip():
                    raise PlanExecutionError("LLM generated empty code")
                
                # Store for potential retry
                self._last_code = cleaned_code
                
                logger.info("Executing code in Docker container...")
                execution_result = self.docker_runner.call_python_script(
                    script=cleaned_code,
                    working_dir=str(self.working_dir),
                    timeout_sec=180  # Increased timeout for complex operations
                )
                
                # Store execution output for potential retry
                self._last_output = execution_result.get('stdout', '') + execution_result.get('stderr', '')
                
                if execution_result['success']:
                    # Parse and validate results
                    result_data = self._parse_execution_result(execution_result, task_data, task_type)
                    
                    # Log successful execution
                    self.execution_history.append({
                        'task_id': task_id,
                        'attempt': attempt + 1,
                        'success': True,
                        'code': cleaned_code,
                        'result': result_data
                    })
                    
                    return True, result_data
                    
                else:
                    # Handle execution failure
                    error_msg = execution_result.get('stderr', execution_result.get('error', 'Unknown execution error'))
                    self._last_error = error_msg
                    
                    logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")
                    
                    if attempt == self.max_retries:
                        # Final attempt failed
                        self.execution_history.append({
                            'task_id': task_id,
                            'attempt': attempt + 1,
                            'success': False,
                            'error': error_msg,
                            'code': cleaned_code
                        })
                        return False, f"All retry attempts failed. Final error: {error_msg}"
                    
                    # Prepare for retry
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
            except Exception as e:
                error_msg = f"Unexpected error in attempt {attempt + 1}: {str(e)}\n{traceback.format_exc()}"
                self._last_error = error_msg
                logger.error(error_msg)
                
                if attempt == self.max_retries:
                    return False, error_msg
                
                time.sleep(2 ** attempt)
        
        return False, "Maximum retry attempts exceeded"

    def _parse_execution_result(self, execution_result: Dict[str, Any], 
                              task_data: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """
        Parse and validate execution results.
        
        Args:
            execution_result: Result from Docker execution
            task_data: Original task configuration
            task_type: Type of task for context
            
        Returns:
            Parsed and validated result data
        """
        stdout = execution_result.get('stdout', '').strip()
        stderr = execution_result.get('stderr', '').strip()
        
        # Basic validation
        if not stdout and stderr:
            logger.warning(f"No stdout but stderr present: {stderr}")
        
        # For analysis tasks, try to extract the final result
        if task_type == "analysis":
            # Look for result indicators in stdout
            lines = stdout.split('\n')
            final_result = None
            
            # Try to find result markers or last meaningful line
            for line in reversed(lines):
                line = line.strip()
                if line and not line.startswith('DEBUG:') and not line.startswith('INFO:'):
                    final_result = line
                    break
            
            return {
                'stdout': stdout,
                'stderr': stderr,
                'final_result': final_result,
                'raw_execution': execution_result
            }
        
        return {
            'stdout': stdout,
            'stderr': stderr,
            'raw_execution': execution_result
        }

    def execute_plan(self) -> List[Any]:
        """
        Execute the complete plan from start to finish.
        
        Returns:
            Final results in the format specified by the plan
        """
        try:
            # Load and validate plan
            logger.info("Loading and validating plan...")
            self.plan_data = self.load_and_validate_plan()
            
            # Execute sourcing phase
            sourcing_result = self.execute_sourcing_plan()
            available_variables = sourcing_result['available_variables']
            
            # Execute analysis phase
            analysis_results = self.execute_analysis_plan(available_variables)
            
            # Final validation and formatting
            expected_format = self.plan_data['final_output_format']
            if expected_format.get('format') == 'array_of_strings':
                # Ensure all results are strings
                formatted_results = []
                for result in analysis_results:
                    if isinstance(result, str):
                        formatted_results.append(result)
                    else:
                        formatted_results.append(str(result))
                analysis_results = formatted_results
            
            logger.info("Plan execution completed successfully!")
            logger.info(f"Final results: {len(analysis_results)} items")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Plan execution failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise PlanExecutionError(f"Plan execution failed: {e}")

    def generate_execution_report(self) -> str:
        """Generate a comprehensive execution report."""
        report = []
        report.append("=== PLAN EXECUTION REPORT ===")
        report.append(f"Plan file: {self.plan_path}")
        report.append(f"Working directory: {self.working_dir}")
        report.append(f"Execution timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total execution history entries: {len(self.execution_history)}")
        report.append("")
        
        # Summarize execution history
        for entry in self.execution_history:
            status = "✓ SUCCESS" if entry['success'] else "✗ FAILURE"
            report.append(f"Task {entry['task_id']} (Attempt {entry['attempt']}): {status}")
            if not entry['success'] and 'error' in entry:
                report.append(f"  Error: {entry['error'][:200]}...")
        
        return "\n".join(report)


def main():
    """Main execution function with command line interface."""
    if len(sys.argv) < 2:
        print("Usage: python execute_plan.py <plan.json> [working_directory]")
        sys.exit(1)
    
    plan_path = sys.argv[1]
    working_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        # Initialize executor
        executor = PlanExecutor(plan_path, working_dir)
        
        # Execute plan
        results = executor.execute_plan()
        
        # Display results
        print("\n" + "="*60)
        print("EXECUTION COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Results ({len(results)} items):")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result}")
        
        # Save results to file
        results_file = executor.working_dir / "execution_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'execution_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'plan_file': str(plan_path),
                'results': results,
                'execution_history': executor.execution_history
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {results_file}")
        
        # Generate and save report
        report = executor.generate_execution_report()
        report_file = executor.working_dir / "execution_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Execution report saved to: {report_file}")
        
    except PlanExecutionError as e:
        logger.error(f"Plan execution error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()