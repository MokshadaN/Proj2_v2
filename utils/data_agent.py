import pandas as pd
import duckdb
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

class DataAgent:
    def __init__(self):
        # Optional: shared DuckDB instance for large file handling
        self.db = duckdb.connect()
        self.max_workers = 4

    def source_data(self, sourcing_plan: dict, uploaded_files: dict) -> dict:
        source_path = sourcing_plan["data_source"]["url"]
        source_type = sourcing_plan["data_source"]["type"]
        file_size = os.path.getsize(source_path) if os.path.exists(source_path) else 0

        # Handle small structured files with Pandas
        if source_type in ["csv", "json", "excel", "parquet"] and file_size < 200 * 1024 * 1024:  # 200MB
            df = self._load_basic_file(source_path, source_type)
            return {
                "type": "pandas",
                "handle": df,
                "sample": df.head(100),
                "schema": df.dtypes.astype(str).to_dict()
            }

        # Handle large structured files with DuckDB
        elif source_type in ["csv", "json", "parquet", "excel"]:
            script = self._get_llm_duckdb_query_script(sourcing_plan)
            self.db.execute(script)
            return {
                "type": "duckdb",
                "handle": self.db,
                "view_name": "raw_data",
                "query_template": "SELECT * FROM raw_data WHERE ..."
            }

        # Handle unknown/unstructured sources using LLM-generated Python code
        else:
            code = self._get_llm_scraper_code(sourcing_plan)
            local_env = {"pd": pd, "requests": __import__("requests"), "os": os}
            exec(code, {}, local_env)
            df = local_env.get("df")
            if df is None:
                raise RuntimeError("LLM code did not return a DataFrame named 'df'")
            return {
                "type": "pandas",
                "handle": df,
                "sample": df.head(100),
                "schema": df.dtypes.astype(str).to_dict()
            }
    def source_data_parallel(self, sourcing_plans: list, uploaded_files: dict = None) -> list:
        """
        Accepts a list of sourcing_plan dicts.
        Runs source_data() concurrently and returns list of results.
        """
        results = [None] * len(sourcing_plans)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Map futures to plan indexes so we can store results in order
            future_to_index = {
                executor.submit(self.source_data, plan, uploaded_files): idx
                for idx, plan in enumerate(sourcing_plans)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = {"error": str(e)}
        return results

    def _load_basic_file(self, path: str, filetype: str) -> pd.DataFrame:
        if filetype == "csv":
            return pd.read_csv(path)
        elif filetype == "json":
            return pd.read_json(path)
        elif filetype == "excel":
            return pd.read_excel(path)
        elif filetype == "parquet":
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported basic file type: {filetype}")

    def _get_llm_duckdb_query_script(self, sourcing_plan: dict) -> str:
        """
        Dummy LLM call that returns an efficient DuckDB script for filtering large files.
        """
        file_path = sourcing_plan["data_source"]["url"]
        query = f"""
        CREATE OR REPLACE VIEW raw_data AS
        SELECT * FROM '{file_path}'
        WHERE revenue > 1000000000  -- This would be generated from extraction_params
        """
        return query

    def _get_llm_scraper_code(self, sourcing_plan: dict) -> str:
        """
        Dummy LLM call that returns a scraping or API-fetching script.
        It must return a DataFrame named `df`.
        """
        return """
        import requests
        from bs4 import BeautifulSoup
        import pandas as pd

        url = 'https://example.com/data'
        html = requests.get(url).text
        soup = BeautifulSoup(html, 'html.parser')

        rows = []
        for tr in soup.select('table tr'):
            cols = tr.find_all('td')
            if len(cols) == 3:
                rows.append({'name': cols[0].text, 'revenue': cols[2].text})

        df = pd.DataFrame(rows)
        """

