from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pathlib
import os
from typing import List, Any
import json
import time

from llm_calls.gemini_llm import call_gemini_llm

from utils.planner_agent import PlannerAgent
from utils.data_agent import DataAgent
from utils.analysis_agent import AnalysisAgent
from utils.prompts import PromptManager
import os

WORKDIR = "./workdir"
os.makedirs(WORKDIR, exist_ok=True) 
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize agents
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

planner_agent = PlannerAgent()
data_agent = DataAgent()
analysis_agent = AnalysisAgent()

def format_final_response(results, output_format, custom_format_description=None, max_retries=3):
    """
    Formats results via Gemini LLM into valid JSON with a feedback loop if parsing fails.
    """
    retries = 0
    error_message = None
    llm_output = None

    while retries < max_retries:
        # Step 1: Build prompt
        if retries == 0:
            system_prompt, user_prompt = PromptManager.formatting_prompt(
                results=results,
                output_format=output_format,
                custom_format_description=custom_format_description
            )
        else:
            # Step 2: Feedback loop prompt
            system_prompt = """
            You previously returned JSON that failed to parse.
            Correct the JSON so that it is valid.
            No markdown, no backticks, no explanations.
            """
            user_prompt = f"""
            Previous output:
            {llm_output}

            Error:
            {error_message}

            Return only valid JSON.
            """

        # Step 3: Call Gemini
        print(f"Gemini API call (attempt {retries+1})...")
        llm_output = call_gemini_llm(system_prompt, user_prompt).strip()

        # Step 4: Try parsing
        try:
            parsed = json.loads(llm_output)
            return parsed  # ✅ Success
        except json.JSONDecodeError as e:
            error_message = str(e)
            retries += 1
            print(f"JSON parsing failed: {error_message}")
    
    # If all retries fail, return last output
    return {
        "formatted_text": llm_output,
        "note": f"Failed to produce valid JSON after {max_retries} attempts"
    }

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
        if hasattr(param_value, 'filename') and param_value.filename:
            file = param_value
            save_path = os.path.join(UPLOAD_FOLDER, pathlib.Path(file.filename).name)
            
            with open(save_path, "wb") as buffer:
                buffer.write(await file.read())
            
            if param_name == 'questions.txt':
                with open(save_path, 'r', encoding='utf-8') as f:
                    questions_content = f.read()
            else:
                uploaded_files_info.append({"filename": file.filename, "param_name": param_name})
                uploaded_files_paths[param_name] = save_path
            uploaded_files_str = "\n".join([
                f"{file['param_name']}: {file['filename']}" for file in uploaded_files_info
            ])
        
    if not questions_content:
        raise HTTPException(status_code=400, detail="questions.txt file not found in the request.")

    try:
        # Step 2: PLAN
        print("Step 2: Generating plan...")
        full_plan = planner_agent.create_plan(questions_content, uploaded_files_str)
        sourcing_plan = full_plan.get('sourcing_plan')
        analysis_plan = full_plan.get('analysis_plan')
        final_output_format = full_plan.get('final_output_format', 'json_array')
        with open("plan.json", "w", encoding="utf-8") as f:
            json.dump(full_plan, f, indent=4)

        if not sourcing_plan or not analysis_plan:
            raise ValueError("PlannerAgent did not return a valid plan.")
        print("Step 2 Result : Plan generated successfully")
        print(json.dumps(full_plan, indent=4)[:50])
        # Step 3: DATA
        print("Step 3: Running DataAgent...")
        data_handle = data_agent.source_data_parallel(sourcing_plan,uploaded_files_paths)
        print("Step 3 Result : Data Sourced successfully",data_handle)

        # # Step 4: ANALYSIS
        print("Step 4: Running AnalysisAgent...")
        results = analysis_agent.run_analysis(analysis_plan, data_handle)
        print("Step 4 Result : Analysis completed successfully")
        print(results)
        

        # # Step 5: FORMAT OUTPUT
        # formatted_output = format_final_response(results, final_output_format)
        formatted_output = "Text"

        end_time = time.time()
        print(f"✅ Full pipeline executed in {end_time - start_time:.2f} seconds.")

        return JSONResponse(content=formatted_output, status_code=200)

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
