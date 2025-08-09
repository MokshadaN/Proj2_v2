# /// script
# dependencies = ["fastapi", "uvicorn", "python-multipart","google-genai","pydantic"]
# ///

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn
import pathlib
import os
import uvicorn
from google import genai
from google.genai import types
import json
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel
from enum import Enum

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

class ToolType(str, Enum):
    DATA_SOURCING = "data_sourcing"
    DATA_PREPARATION = "data_preparation"
    DATA_ANALYSIS = "data_analysis"
    DATA_VISUALIZATION = "data_visualization"

class SourceType(str, Enum):
    CSV = "csv"
    JSON = "json"
    HTML = "html"
    PARQUET = "parquet"
    SQL = "sql"
    API = "api"
    EXCEL = "excel"
    PDF = "pdf"
    OTHER = "other"

class OutputFormat(str, Enum):
    PANDAS_DATAFRAME = "pandas_dataframe"
    BASE64_WEBP = "base64_webp"
    BASE64_PNG = "base64_png"
    STRING = "string"
    FLOAT = "float"
    INTEGER = "integer"
    JSON = "json"
    CSV_FILE = "csv_file"
    PARQUET_FILE = "parquet_file"
    HTML = "html"
    WEBP = "webp"
    PNG = "png"
    JPEG ="jpeg"
    OTHER = "other"

# Base Step Structure
class BaseStep(BaseModel):
    step: int
    tool: ToolType
    description: str
    depends_on: List[int] = []
    expected_format: OutputFormat
    error_handling: Dict[str, Any] = {}
    validation_rules: List[str] = []

def run_planner_agent_v1(task_description, metadata, files):
    system_instruction = f"""You are an expert data analyst assistant that creates structured execution plans.

    Your task is to break down a user's data analysis request into an optimized, structured, and sequential plan using FIXED STEP STRUCTURES.

    **MANDATORY STEP STRUCTURES:**

    **1. DATA_SOURCING STEP:**
    {{
        "step": <number>,
        "tool": "data_sourcing",
        "description": "<clear description>",
        "input": {{}}, // Usually empty for first step
        "output": {{
            "variable_name": "<variable_name>",
            "file_path": "<persistence_path>",
            "schema": {{"columns": [...], "dtypes": {{...}}}}
        }},
        "data_source": {{
            "url": "<source_url_or_path>",
            "type": "<source_type>"
        }},
        "codes" : {{
            code given in the metadata that could be run 
            }}
        "source_type": "<{[t.value for t in SourceType]}>",
        "extraction_params": {{
            // Source-specific parameters (let them be very specific no jargon no assumption)
        }},
        "expected_format": "pandas_dataframe",
        "error_handling": {{
            "missing_file": "raise_exception",
            "encoding_error": "try_alternative_encodings"
        }},
        "validation_rules": ["check_required_columns", "validate_data_types"],
        "questions" :[list of all questions to answer]
        "depends_on": []
    }}

    **2. DATA_PREPARATION STEP:**
    {{
        "step": <number>,
        "tool": "data_preparation", 
        "description": "<clear description>",
        "input": {{
            "dataframe": "<input_dataframe_name>",
            "columns": [<list_of_columns>]
        }},
        "output": {{
            "variable_name": "<output_variable_name>",
            "file_path": "<optional_persistence_path>",
            "transformations_applied": [<list_of_transformations>]
        }},
        "operations": [
            {{"type": "<operation_type>", "columns": [...], "parameters": {{...}}}}
        ],
        "expected_format": "pandas_dataframe",
        "error_handling": {{
            "invalid_data": "<strategy>",
            "missing_values": "<strategy>"
        }},
        "validation_rules": [<validation_checks>],
        "depends_on": [<prerequisite_steps>]
    }}

    **3. DATA_ANALYSIS STEP:**
    {{
        "step": <number>,
        "tool": "data_analysis",
        "description": "<clear description>",
        "input": {{
            "dataframe": "<input_dataframe_name>",
            "columns": [<analysis_columns>]
        }},
        "output": {{
            "variable_name": "<result_variable_name>",
            "metrics": {{
                "<metric_name>": "<data_type>"
            }},
            "file_path": "<optional_results_file>"
        }},
        "analysis_type": "<correlation|regression|aggregation|statistical_test|etc>",
        "parameters": {{
            // Analysis-specific parameters
        }},
        "expected_format": "<json|float|string|pandas_dataframe>",
        "error_handling": {{
            "calculation_error": "<strategy>",
            "insufficient_data": "<strategy>"
        }},
        "validation_rules": [<validation_checks>],
        "depends_on": [<prerequisite_steps>]
    }}

    **4. DATA_VISUALIZATION STEP:**
    {{
        "step": <number>,
        "tool": "data_visualization",
        "description": "<clear description>",
        "input": {{
            "dataframe": "<input_dataframe_name>",
            "x_column": "<x_axis_column>",
            "y_column": "<y_axis_column>"
        }},
        "output": {{
            "variable_name": "<chart_variable_name>",
            "file_path": "<chart_file_path>",
            "encoding": "base64_webp",
            "dimensions": {{"width": 800, "height": 600}}
        }},
        "plot_type": "<bar_chart|line_plot|scatter_plot|heatmap|etc>",
        "visual_params": {{
            "title": "<chart_title>",
            "x_axis": {{"label": "<label>", "format": "<format>"}},
            "y_axis": {{"label": "<label>", "format": "<format>"}},
            "color": {{"column": "<color_column>", "palette": "<palette>"}},
            "legend": {{"show": true, "position": "<position>"}}
        }},
        "expected_format": "base64_webp",
        "error_handling": {{
            "missing_data": "show_empty_chart_message",
            "rendering_error": "fallback_to_table"
        }},
        "validation_rules": ["ensure_data_exists", "validate_column_types"],
        "depends_on": [<prerequisite_steps>]
    }}

    **CRITICAL REQUIREMENTS:**
    1. Every step MUST follow the exact structure above for its tool type
    2. Use METADATA to populate data_source URLs, column names, and data types exactly
    3. Each step must be independently executable with all required context
    4. Group data sourcing steps when possible to minimize expensive operations try to keep them down to 1 for each source with the required columns and conditions
    5. Include comprehensive error handling and validation rules
    6. Ensure output names are consistent across steps for proper dependency tracking
    7. If the metadata or the question conatins any script add it in that particular steps description and an extra field (manadatory) also do not mention any method that requires authentication (like s3)
    8. The final step should format the answer according to the task requirements

    Return ONLY a JSON array of steps following these exact structures.
    """

    user_prompt = f"""Create a structured execution plan using the fixed step structures.

    METADATA:
    {metadata}

    TASK:
    {task_description}

    FILES:
    {files}

    Return a JSON array of steps following the exact structures specified in the system instructions.
    """

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction, 
            response_mime_type="application/json"
        ),
        contents=user_prompt
    )

    try:
        parsed_plan = json.loads(response.text)
    except json.JSONDecodeError as e:
        print("❌ Failed to parse JSON from LLM response")
        print(f"LLM Response Text: {response.text}")
        raise e

    # Validate and inject dependencies
    validated_plan = validate_and_inject_dependencies(parsed_plan)

    with open("plan.json", "w") as f:
        json.dump(validated_plan, f, indent=2)

    print("✅ Saved structured plan.json with dependencies successfully.")
    return validated_plan

def run_planner_agent(task_description,metadata,files):
    system_instruction = f"""You are an expert data analyst assistant that creates structured execution plans.

    Your task is to break down a user's data analysis request into an optimized, structured, and sequential plan. For efficiency, you must combine the data retrieval and initial data cleaning/transformation into a single, comprehensive 'data_sourcing' step.

    **MANDATORY STEP STRUCTURES:**

    **1. DATA_SOURCING STEP ALONG WITH CLEANING:**
    {{
        "step": <number>,
        "tool": "data_sourcing",
        "description": "<clear description of data retrieval and initial preparation>",
        "input": {{}}, // Usually empty for the first step
        "output": {{
            "variable_name": "<variable_name>",
            "file_path": "<persistence_path>",
            "schema": {{"columns": [...], "dtypes": {{...}}}},
            "transformations_applied": [<list_of_transformations>]
        }},
        "data_source": {{
            "url": "<source_url_or_path>",
            "type": "<source_type>"
        }},
        "codes" : {{
            "code": "<code_given_in_metadata>"
        }},
        "source_type": "<{[t.value for t in SourceType]}>",
        "extraction_params": {{
            // Source-specific parameters
        }},
        "operations": [
            {{"type": "<operation_type>", "columns": [...], "parameters": {{...}}}}
        ],
        "expected_format": "pandas_dataframe",
        "error_handling": {{
            "missing_file": "raise_exception",
            "encoding_error": "try_alternative_encodings",
            "invalid_data": "<strategy_for_missing_values>",
            "type_mismatch": "<strategy_for_type_errors>"
        }},
        "validation_rules": ["check_required_columns", "validate_data_types", "check_for_duplicates"],
        "questions" :[list of all questions to answer],
        "depends_on": []
    }}

    **2. DATA_ANALYSIS STEP:**
    {{
        "step": <number>,
        "tool": "data_analysis",
        "description": "<clear description>",
        "input": {{
            "dataframe": "<input_dataframe_name>",
            "columns": [<analysis_columns>]
        }},
        "output": {{
            "variable_name": "<result_variable_name>",
            "metrics": {{
                "<metric_name>": "<data_type>"
            }},
            "file_path": "<optional_results_file>"
        }},
        "analysis_type": "<correlation|regression|aggregation|statistical_test|etc>",
        "parameters": {{
            // Analysis-specific parameters
        }},
        "expected_format": "<json|float|string|pandas_dataframe>",
        "error_handling": {{
            "calculation_error": "<strategy>",
            "insufficient_data": "<strategy>"
        }},
        "validation_rules": [<validation_checks>],
        "depends_on": [<prerequisite_steps>]
    }}

    **3. DATA_VISUALIZATION STEP:**
    {{
        "step": <number>,
        "tool": "data_visualization",
        "description": "<clear description>",
        "input": {{
            "dataframe": "<input_dataframe_name>",
            "x_column": "<x_axis_column>",
            "y_column": "<y_axis_column>"
        }},
        "output": {{
            "variable_name": "<chart_variable_name>",
            "file_path": "<chart_file_path>",
            "encoding": "base64_webp",
            "dimensions": {{"width": 800, "height": 600}}
        }},
        "plot_type": "<bar_chart|line_plot|scatter_plot|heatmap|etc>",
        "visual_params": {{
            "title": "<chart_title>",
            "x_axis": {{"label": "<label>", "format": "<format>"}},
            "y_axis": {{"label": "<label>", "format": "<format>"}},
            "color": {{"column": "<color_column>", "palette": "<palette>"}},
            "legend": {{"show": true, "position": "<position>"}}
        }},
        "expected_format": "base64_webp",
        "error_handling": {{
            "missing_data": "show_empty_chart_message",
            "rendering_error": "fallback_to_table"
        }},
        "validation_rules": ["ensure_data_exists", "validate_column_types"],
        "depends_on": [<prerequisite_steps>]
    }}

    **CRITICAL REQUIREMENTS:**
    0. **Get the correct tags and complete details for data sourcing as it is the most crucial step in the entire analysis**
    1. The first step of the plan MUST use the `data_sourcing` tool, which now includes both data retrieval and initial cleaning operations.
    2. Every step MUST follow the exact structure above for its tool type.
    3. Use METADATA to populate data_source URLs, column names, and data types exactly.
    4. Each step must be independently executable with all required context.
    5. Ensure output names are consistent across steps for proper dependency tracking.
    6. If the metadata or the question contains any script, add it in that particular step's description and an extra 'codes' field. Do not mention any method that requires authentication (like s3).
    7. The final step should format the answer according to the task requirements.

    Return ONLY a JSON array of steps following these exact structures.
    """

    user_prompt = f"""Create a structured execution plan using the fixed step structures.

    METADATA:
    {metadata}

    TASK:
    {task_description}

    FILES:
    {files}

    Return a JSON array of steps following the exact structures specified in the system instructions.
    """
    user_prompt = f"""Create a structured execution plan using the fixed step structures.

    METADATA:
    {metadata}

    TASK:
    {task_description}

    FILES:
    {files}

    Return a JSON array of steps following the exact structures specified in the system instructions.
    """

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction, 
            response_mime_type="application/json"
        ),
        contents=user_prompt
    )

    try:
        parsed_plan = json.loads(response.text)
    except json.JSONDecodeError as e:
        print("❌ Failed to parse JSON from LLM response")
        print(f"LLM Response Text: {response.text}")
        raise e

    # Validate and inject dependencies
    validated_plan = validate_and_inject_dependencies(parsed_plan)

    with open("plan.json", "w") as f:
        json.dump(validated_plan, f, indent=2)

    print("✅ Saved structured plan.json with dependencies successfully.")
    return validated_plan
   

def validate_and_inject_dependencies(plan):
    """
    Validate step structures and inject dependencies based on input/output relationships.
    """
    output_to_step = {}
    validated_plan = []

    # First pass: validate structures and map outputs
    for step in plan:
        # Validate required fields based on tool type
        tool_type = step.get("tool")
        
        required_fields = {
            "data_sourcing": ["step", "tool", "description", "output", "data_source", "source_type", "expected_format"],
            "data_preparation": ["step", "tool", "description", "input", "output", "operations", "expected_format"],
            "data_analysis": ["step", "tool", "description", "input", "output", "analysis_type", "parameters", "expected_format"],
            "data_visualization": ["step", "tool", "description", "input", "output", "plot_type", "visual_params", "expected_format"]
        }

        if tool_type in required_fields:
            missing_fields = [field for field in required_fields[tool_type] if field not in step]
            if missing_fields:
                print(f"⚠️ Step {step.get('step', 'unknown')} missing required fields: {missing_fields}")

        # Add default fields if missing
        step.setdefault("depends_on", [])
        step.setdefault("error_handling", {})
        step.setdefault("validation_rules", [])

        # Map outputs to step numbers
        if "output" in step:
            output = step["output"]
            for key, value in output.items():
                if isinstance(value, str):
                    output_to_step[value] = step["step"]

        validated_plan.append(step)

    # Second pass: inject dependencies
    for step in validated_plan:
        deps = set()
        
        def find_dependencies(obj):
            if isinstance(obj, str) and obj in output_to_step:
                deps.add(output_to_step[obj])
            elif isinstance(obj, dict):
                for value in obj.values():
                    find_dependencies(value)
            elif isinstance(obj, list):
                for item in obj:
                    find_dependencies(item)

        # Check input fields for dependencies
        if "input" in step:
            find_dependencies(step["input"])

        step["depends_on"] = sorted(list(deps))

    return validated_plan


# def run_planner_agent(task_description, metadata, files):
#     system_instruction = """You are an expert data analyst assistant.

#     Your task is to break down a user's data analysis request into an optimized, structured, and sequential plan. The plan should be suitable for automated execution and code generation, and should be in the form of a JSON array of steps.

#     Each step must be formatted as a JSON object and include:

#     - step: A unique identifier for the step (e.g., 1, 2, 3).
#     - tool: One of the following categories — "data_sourcing", "data_preparation", "data_analysis", "data_visualization".
#     - description: A human-readable summary of the step's purpose.
#     - input: What is required to perform this step (e.g., file path, columns, filters, dataframe from previous step).
#     - output: What will be produced and its structure.
#     - expected_format: The format of the output (e.g., "pandas_dataframe", "base64_webp", "string", "float", "database", or anything relevant for faster processing).
#     - data_source: A clear specification of which data source is being used in this step. and also the data source url refering .This should be a URL, file name, table name, or other identifier from the METADATA.

#     **Key instructions for planning:**

#     0. Use the provided METADATA section to extract column names, formats, storage paths, and data types.
#     - Apply this information to filter design, date parsing, format assumptions, and downstream analysis.
#     - Use column names and date formats exactly as shown in the metadata or schema.
#     - If multiple data sources are listed (e.g., multiple file paths, URLs, tables), identify the **most relevant source(s)** based on the user's task and explicitly specify which source is being used for each step.
#     - Justify source selection if multiple datasets could be candidates for a given analysis.

#     1. Group related data_sourcing steps whenever possible to minimize repeated expensive operations (e.g., querying S3 Parquet multiple times).
#     and if the data sourcing is required from html also mention the table structure and a sample <table> tag sourced from the website and column data types by getting 2 rows of the table
#     - For each data_sourcing step, store the output as an in-memory pandas DataFrame.
#     - If the dataset is large or needs reuse, also persist it as a Parquet file and include the file path in the output field.

#     2. Ensure each step is independently executable and contains all the context needed for downstream code generation.

#     3. If the task references analysis, explicitly name the analysis type (e.g., "count", "correlation", "regression", "average") and keep all the numbered questions separate.

#     4. For visualization steps, include:
#     - Plot type (e.g., "scatterplot", "bar_chart")
#     - Encoding format (e.g., "base64_webp")

#     5. Output the final plan as a JSON array of steps, suitable for downstream automation.

#     6. Include the answer format that is required as the task description

#     7. Let the last step be the format of the answer to the question in the format specified in the task description. PLease make sure the answer is in the format specified in the task description.

#     8. If the task includes URL-based data extraction (e.g., from HTML web pages), explicitly include a data_sourcing step that:
#     - Specifies the URL(s) to scrape.
#     - Describes the expected table structure(s) on the page, including column names and data types.
#     - Provides a sample HTML <table> snippet extracted from the URL with 2 example rows to clarify the schema for downstream steps.
#     - Explains how this data will be parsed and transformed into a structured format (e.g., pandas DataFrame).

#     9. When multiple URLs or diverse file formats are involved, separate the sourcing steps per format/type and ensure their outputs are clearly named for reuse.

#     10. In planning, explicitly break down multi-part questions into separate steps, each clearly linked to the relevant data processing or analysis operation.

#     11. Prioritize efficiency by grouping data sourcing where possible, but do not sacrifice clarity or step independence.

#     12. Ensure every step includes enough detail for code generation without ambiguity, especially for transformations like date parsing, filtering criteria, and calculation methods.

#     Return only the JSON array, with no additional explanation or formatting.
#     """

#     user_prompt = f"""You will receive two inputs: a dataset metadata section and a task.

#     Use the metadata to inform data sourcing, transformation, analysis, and visualization. Apply schema, data types, date formats, and assumptions from the metadata to improve the quality of the plan.

#     ---
#     METADATA:
#     {metadata}

#     ---
#     TASK:
#     {task_description}

#     ---
#     FILES:
#     {files}

#     ---
#     OUTPUT FORMAT:
#     The output should be a JSON array of steps, suitable for downstream automation.
#     """

#     response = client.models.generate_content(
#         model="gemini-2.5-flash",
#         config=types.GenerateContentConfig(system_instruction=system_instruction, response_mime_type="application/json"),
#         contents=user_prompt
#     )

#     try:
#         parsed_plan = json.loads(response.text)
#     except json.JSONDecodeError as e:
#         print("❌ Failed to parse JSON from LLM response")
#         print(f"LLM Response Text: {response.text}")
#         raise e

#     # 🔁 Inject depends_on tags
#     plan_with_dependencies = inject_dependencies(parsed_plan)

#     with open("plan.json", "w") as f:
#         json.dump(plan_with_dependencies, f, indent=2)

#     print("✅ Saved plan.json with dependencies successfully.")
#     return plan_with_dependencies


# def inject_dependencies(plan):
#     """
#     Analyze plan steps and inject a 'depends_on' list into each step.
#     Handles both string and dict outputs gracefully.
#     """
#     output_to_step = {}

#     # First pass: map outputs to step numbers
#     for step in plan:
#         if "output" in step:
#             output = step["output"]
#             if isinstance(output, dict):
#                 # Extract the first value as output name (since key names vary)
#                 output_name = next(iter(output.values()), None)
#             elif isinstance(output, str):
#                 output_name = output
#             else:
#                 output_name = None

#             if output_name:
#                 output_to_step[output_name] = step["step"]

#     # Second pass: infer dependencies based on input fields
#     for step in plan:
#         deps = set()
#         inputs = step.get("input", {})

#         def collect_deps(value):
#             if isinstance(value, str) and value in output_to_step:
#                 deps.add(output_to_step[value])
#             elif isinstance(value, list):
#                 for item in value:
#                     collect_deps(item)
#             elif isinstance(value, dict):
#                 for v in value.values():
#                     collect_deps(v)

#         collect_deps(inputs)
#         step["depends_on"] = sorted(list(deps))

#     return plan

def get_metadata(task_description):
    system_instruction = """
    You are a metadata extraction assistant that creates structured metadata for data sources.

    Extract detailed metadata from dataset descriptions involving multiple data sources. Return a JSON array with this EXACT structure for each source:

    {
        "url_or_path": "<source_identifier>",
        "source_type": "<csv|json|html|parquet|sql|api|excel|pdf|txt|anyother>",
        "columns": [
            {
                "name": "<column_name>",
                "type": "<data_type>",
                "description": "<optional_description>",
                "sample_values": ["<sample1>", "<sample2>"]
            }
        ],
        "sample_rows": [
            {<sample_row_1>},
            {<sample_row_2>}
        ],
        "format_assumptions": {
            "date_format": "<format_string>",
            "number_format": "<format_string>",
            "encoding": "<encoding>",
            "separator": "<separator>"
        },
        "storage_info": {
            "size": "<estimated_size>",
            "record_count": "<estimated_count>",
            "compression": "<compression_type>"
        },
        "extraction_parameters": {
            "headers": {<http_headers>},
            "authentication": "<auth_method>",
            "query_parameters": {<query_params>}
        },
        "dom_structure": {
            "html_selector" :<selector>,
            "attributes": {
                "class": "<class_attribute>",
                "id": "<id_attribute>",
                "other_attributes": "<any_other_attributes>"
            },
            "html_table_sample" : <sample_html>
        },
        "quality_notes": [
            "<data_quality_observation>"
        ],
        "any other useful information or codes given " : [
            {"codes" : "observation"}
        ]
    }

    For HTML sources, include detailed DOM structure information.
    For HTML tables , correctly include the table selector, sample HTML snippet, and row numbers for headers and data as they will be used for sourcing 
    For APIs, include authentication and parameter requirements.
    For files, include format-specific parsing parameters.

    Return only the JSON array with no additional text.
    """

    user_prompt = f"""Extract structured metadata for all data sources mentioned in this task:

    {task_description}

    **MAKE IT AS USEFUL AND EFFICIENT FOR DATA SOURCING AND ANALYSIS AS POSSIBLE. **

    Return a JSON array of metadata objects following the exact structure specified."""
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(system_instruction=system_instruction),
        contents=user_prompt
    )
    return response.text


