from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import pathlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Dict, Any, Set, List

from llm_calls.gemini_llm import call_gemini_llm

from utils.planner_agent import PlannerAgent
from utils.data_agent import DataAgent
from utils.analysis_agent import AnalysisAgent
from utils.prompts import PromptManager

WORKDIR = "./workdir"
os.makedirs(WORKDIR, exist_ok=True)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = FastAPI()

# Placeholder DataAgent
class DataAgent:
    def source_data(self, plan_steps, uploaded_files):
        # Simulate data handle: a dict with dummy info
        print("DataAgent: Sourcing data for step 1")
        return {"type": "pandas_dataframe", "variable_name": "students_df", "data": "dummy_data"}

# Placeholder AnalysisAgent
class AnalysisAgent:
    def run_analysis(self, plan_steps, data_handle):
        # Simulate running analysis on step 2 and 3
        print("AnalysisAgent: Running analysis for steps 2 and 3")
        return {"analysis_result": "dummy_analysis_output"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Initialize agents
planner_agent = PlannerAgent()
data_agent = DataAgent()
analysis_agent = AnalysisAgent()
# If you have visualization agent, init it here

def format_final_response(results, output_format, custom_format_description=None, max_retries=3):
    # Same as before, omitted here for brevity
    pass

def get_ready_steps(plan_steps, completed_steps: Set[int]) -> List[Dict]:
    """Return steps whose dependencies are all completed and not yet executed."""
    ready = []
    for step in plan_steps:
        if step["step"] in completed_steps:
            continue
        if all(dep in completed_steps for dep in step.get("depends_on", [])):
            ready.append(step)
    return ready

@app.get("/")
def read_root():
    return {"message": "Data Analyst Agent API"}

@app.post("/api")
async def handle_analysis_request(request: Request):
    start_time = time.time()

    try:
        form_data = await request.form()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid form data: {str(e)}")

    uploaded_files_info = []
    questions_content = ""
    uploaded_files_paths = {}

    for param_name, param_value in form_data.items():
        if hasattr(param_value, "filename") and param_value.filename:
            file = param_value
            save_path = os.path.join(UPLOAD_FOLDER, pathlib.Path(file.filename).name)

            with open(save_path, "wb") as buffer:
                buffer.write(await file.read())

            if param_name == "questions.txt":
                with open(save_path, "r", encoding="utf-8") as f:
                    questions_content = f.read()
            else:
                uploaded_files_info.append({"filename": file.filename, "param_name": param_name})
                uploaded_files_paths[param_name] = save_path

    if not questions_content:
        raise HTTPException(status_code=400, detail="questions.txt file not found in the request.")

    try:
        # Step 1: PlannerAgent - generate full plan JSON
        print("Step 1: Generating plan with PlannerAgent...")
        # full_plan = planner_agent.create_plan(questions_content, "\n".join([f"{f['param_name']}: {f['filename']}" for f in uploaded_files_info]))
        # plan_steps = full_plan.get("plan", None)
        with open("plan.json","r") as f:
            full_plan = json.load(f)
        plan_steps = full_plan
        # final_output_format = full_plan.get("final_output_format", "json_array")

        if plan_steps is None or not isinstance(plan_steps, list):
            raise ValueError("PlannerAgent did not return a valid 'plan' list.")

        print(f"Plan with {len(plan_steps)} steps received.")

        # Shared storage for step outputs keyed by step number
        step_outputs: Dict[int, Any] = {}

        completed_steps: Set[int] = set()
        in_progress_steps: Set[int] = set()

        max_workers = min(5, len(plan_steps))  # limit concurrency

        # Executor for parallel step execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            futures = {}

            def submit_step(step):
                tool = step.get("tool")
                step_num = step.get("step")
                print(f"Submitting step {step_num} (tool={tool})...")

                if tool == "planner":
                    # Rare case if plan has planner step inside (usually only one plan step)
                    future = executor.submit(planner_agent.create_plan, questions_content, "\n".join([f"{f['param_name']}: {f['filename']}" for f in uploaded_files_info]))
                elif tool == "data_sourcing" or tool == "data_source" or tool == "data_sourcing": 
                    # Pass the step dict and uploaded files
                    future = executor.submit(data_agent.source_data, step, uploaded_files_paths)
                elif tool == "data_analysis" or tool == "analysis":
                    # For analysis, pass step dict and data handles
                    # Usually depends on previous data_sourcing steps output
                    # Pass step_outputs of dependencies or single dependency data_handle
                    # We'll gather all dependency outputs as context if needed
                    input_data_handles = [step_outputs[dep] for dep in step.get("depends_on", []) if dep in step_outputs]
                    # For simplicity, pass first if single input, else list
                    data_handle = input_data_handles[0] if len(input_data_handles) == 1 else input_data_handles
                    future = executor.submit(analysis_agent.run_analysis, step, data_handle)
                elif tool == "data_visualization" or tool == "visualization":
                    # Similar logic to analysis
                    input_data_handles = [step_outputs[dep] for dep in step.get("depends_on", []) if dep in step_outputs]
                    data_handle = input_data_handles[0] if len(input_data_handles) == 1 else input_data_handles
                    # future = executor.submit(visualization_agent.run_visualization, step, data_handle)
                    # For now, just dummy placeholder
                    future = executor.submit(lambda s, d: {"status": "visualization done", "step": s["step"]}, step, data_handle)
                else:
                    # Unknown tool: raise error or skip
                    future = executor.submit(lambda: {"error": f"Unknown tool {tool} in step {step_num}"})

                return future

            # Loop until all steps completed
            while len(completed_steps) < len(plan_steps):

                ready_steps = get_ready_steps(plan_steps, completed_steps.union(in_progress_steps))

                if not ready_steps and len(in_progress_steps) == 0:
                    # No steps ready, no steps running => deadlock or done
                    if len(completed_steps) < len(plan_steps):
                        raise RuntimeError("Deadlock detected: steps remain but no progress possible.")
                    break

                # Submit all ready steps
                for step in ready_steps:
                    step_num = step["step"]
                    if step_num not in futures:
                        future = submit_step(step)
                        futures[future] = step
                        in_progress_steps.add(step_num)

                # Wait for any future to complete
                done_futures, _ = as_completed(futures.keys(), timeout=5), None

                # Process completed futures
                to_remove = []
                for fut in list(futures.keys()):
                    if fut.done():
                        step = futures[fut]
                        step_num = step["step"]
                        try:
                            result = fut.result()
                            print(f"Step {step_num} completed successfully.")
                            step_outputs[step_num] = result
                        except Exception as e:
                            print(f"Step {step_num} failed with error: {e}")
                            step_outputs[step_num] = {"error": str(e)}
                        completed_steps.add(step_num)
                        in_progress_steps.remove(step_num)
                        to_remove.append(fut)

                for fut in to_remove:
                    futures.pop(fut)

                time.sleep(0.1)  # small delay to avoid busy wait

        # After all steps done, aggregate outputs as needed
        # Here, as example, return the last step output or entire outputs dict
        final_result = step_outputs.get(plan_steps[-1]["step"], step_outputs)

        # Format final output if needed (assuming final_result is the analysis results)
        # formatted_output = format_final_response(final_result, final_output_format)
        formatted_output = final_result
        formatted_output = "text"
        end_time = time.time()
        print(f"✅ Full DAG pipeline executed in {end_time - start_time:.2f} seconds.")

        return JSONResponse(content=formatted_output, status_code=200)

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
