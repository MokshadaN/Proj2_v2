import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import anthropic

class SimpleAnthropicAnalysisAgent:
    def __init__(self, api_key: str, max_workers: int = 4, max_retries: int = 3):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"  # Claude with code exec support
        self.max_workers = max_workers
        self.max_retries = max_retries

    def create_analysis_prompt(self, task: Dict[str, Any], context: Dict[str, Any]) -> str:
        output_format = task.get('output_format', 'json')

        prompt = f"""
You have access to a Python code execution environment. Please solve this data analysis task step by step.

TASK: {task.get('question', 'Analyze the data')}

DATA INFORMATION:
- File path: {context.get('data_path', 'data.csv')}
- Data type: {context.get('data_type', 'csv')}
- Schema: {json.dumps(context.get('schema', {}), indent=2)}
- Sample data: {json.dumps(context.get('data_sample', []), indent=2)}

REQUIREMENTS:
- Task ID: {task.get('task_id')}
- Expected output format: {output_format}

Instructions:
- Use Python to perform the analysis.
- Print **only** the final result on the **last line** in the exact format requested.

"""

        if output_format == 'number':
            prompt += "Print a single number (int or float), e.g., 42 or 3.14.\n"
        elif output_format == 'string':
            prompt += 'Print a single string surrounded by quotes, e.g., "Math_101".\n'
        elif output_format in ['json', 'json_object', 'json_array']:
            prompt += 'Print valid JSON, e.g., {"unique_classes": 12, "most_populated_class": "Math_101"}.\n'

        prompt += """
CRITICAL:
- Your final output **must be parsable** as the specified format.
- Do not print anything else after the final output.
"""

        return prompt

    def execute_task(self, task: Dict[str, Any], data_handle: Dict[str, Any]) -> Dict[str, Any]:
        task_id = task.get("task_id")
        question_id = task.get("qid", task.get("question_id"))
        context = {
            "data_type": data_handle.get("type", "csv"),
            "schema": data_handle.get("schema", {}),
            "data_sample": data_handle.get("sample", []),
            "data_path": data_handle.get("path", "data.csv"),
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                prompt = self.create_analysis_prompt(task, context)
                if last_error:
                    prompt += f"\n\nPrevious error:\n{last_error}\nPlease fix and try again."

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                    tools=[{
                        "type": "code_execution_20250522",
                        "name": "code_execution"
                    }],
                    extra_headers={"anthropic-beta": "code-execution-2025-05-22"}
                )
                
                answer = self.extract_answer_from_response(response, task)
                if answer is not None:
                    return {
                        "task_id": task_id,
                        "question_id": question_id,
                        "answer": answer,
                        "error": None,
                        "attempt": attempt + 1
                    }
                else:
                    last_error = "Could not parse output from response."
            except Exception as e:
                last_error = f"API call failed: {e}"

        return {
            "task_id": task_id,
            "question_id": question_id,
            "answer": None,
            "error": f"Max retries reached. Last error: {last_error}",
            "attempt": self.max_retries
        }

    def extract_answer_from_response(self, response: Any, task: Dict[str, Any]) -> Any:
        """Extract final output by taking last non-empty line and parsing as per output_format"""
        try:
            # Collect all text content from response
            all_text = ""
            for block in response.content:
                if hasattr(block, 'type') and block.type == 'text':
                    all_text += block.text + "\n"
                elif hasattr(block, 'type') and block.type == 'tool_result':
                    # tool_result might have execution output
                    if hasattr(block, 'content'):
                        if isinstance(block.content, list):
                            for item in block.content:
                                if hasattr(item, 'type') and item.type == 'text':
                                    all_text += item.text + "\n"
                        else:
                            all_text += str(block.content) + "\n"

            # Extract last non-empty line
            lines = [line.strip() for line in all_text.strip().splitlines() if line.strip()]
            if not lines:
                return None
            last_line = lines[-1]

            output_format = task.get('output_format', 'json')

            # Parse according to output_format
            if output_format == 'number':
                # Try to convert to float or int
                try:
                    return int(last_line)
                except ValueError:
                    return float(last_line)
            elif output_format == 'string':
                # Strip quotes if present
                if (last_line.startswith('"') and last_line.endswith('"')) or (last_line.startswith("'") and last_line.endswith("'")):
                    return last_line[1:-1]
                return last_line
            elif output_format in ['json', 'json_object', 'json_array']:
                return json.loads(last_line)
            else:
                return last_line

        except Exception as e:
            print(f"Error parsing response output: {e}")
            return None

    def run_analysis(self, analysis_plan: List[Dict[str, Any]], data_handle: Dict[str, Any]) -> List[Dict[str, Any]]:
        print(f"Running {len(analysis_plan)} analysis tasks with max_workers={self.max_workers}")
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {executor.submit(self.execute_task, task, data_handle): task for task in analysis_plan}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    res = future.result()
                    results.append(res)
                    if res.get('error'):
                        print(f"Task {res['task_id']} failed: {res['error']}")
                    else:
                        print(f"Task {res['task_id']} succeeded")
                except Exception as e:
                    results.append({
                        "task_id": task.get('task_id'),
                        "question_id": task.get('qid', task.get('question_id')),
                        "answer": None,
                        "error": f"Future failed: {e}",
                        "traceback": traceback.format_exc()
                    })
        results.sort(key=lambda r: r.get('task_id', 0))
        print(f"Analysis complete. Success: {len([r for r in results if not r.get('error')])}, Failed: {len([r for r in results if r.get('error')])}")
        return results

# Usage example
if __name__ == "__main__":
    agent = SimpleAnthropicAnalysisAgent(api_key="YOUR_API_KEY", max_workers=3, max_retries=2)
    analysis_plan = [
        {
            "task_id": 1,
            "qid": "q1",
            "question": "How many unique classes are there?",
            "output_format": "number"
        },
        {
            "task_id": 2,
            "qid": "q2",
            "question": "Which class has the highest number of students?",
            "output_format": "string"
        }
    ]
    data_handle = {
        "type": "csv",
        "path": "q-fastapi.csv",
        "schema": {"studentId": "int64", "class": "object"},
        "sample": [
            {"studentId": 101, "class": "Math_101"},
            {"studentId": 102, "class": "Physics_202"}
        ]
    }

    results = agent.run_analysis(analysis_plan, data_handle)
    for r in results:
        print(r)
