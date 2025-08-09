curl "http://127.0.0.1:8000/api" -F "questions.txt=@question_m.txt" -F "q-fastapi.csv=@q-fastapi.csv"
curl "http://127.0.0.1:8000/api" -F "questions.txt=@question_csv.txt"

 curl "http://127.0.0.1:8000/api" -F "questions.txt=@question_csv.txt" -F "q-fastapi.csv=@q-fastapi.csv"

1. All kinds of source URLs that can be in questions.txt
From your description, questions.txt can contain one or many sources, mixed with plain text. The extraction must support structured and unstructured formats.

URL types to expect (grouped):

Category	Examples	Notes
Web HTML pages	https://en.wikipedia.org/wiki/..., https://example.com/table	Likely requires HTML parsing (BeautifulSoup, lxml, pandas.read_html)
Direct file links	https://example.com/data.csv, .json, .parquet, .tsv, .xlsx	Can be streamed directly to DuckDB or Pandas without storing full file in RAM
API endpoints	https://api.example.com/v1/data?key=...	Usually returns JSON, sometimes paginated; may require authentication
Cloud object storage (public)	https://bucket.s3.amazonaws.com/path/file.parquet, gs://bucket/path/file.csv, azure://container/blob	Often large; use partial fetch / query pushdown
Database URIs	postgresql://user:pass@host:port/dbname, mysql://...	Requires SQL connection; only pass queries to fetch needed columns/rows
Data portals	https://data.gov/..., https://opendata.city/api/...	May be HTML index pages or APIs with datasets
Special sources	ftp://..., file:///...	Only if explicitly enabled for local/FTP reads

Multi-URL case:
We should scan all lines of questions.txt with regex like:

python
Copy
Edit
re.findall(r'(https?://\S+|ftp://\S+|gs://\S+|s3://\S+)', text)
and then classify by extension / scheme.

2. All kinds of file attachments in curl request
Your /api/ endpoint can receive zero, one, or many files via multipart form-data.

Likely attachment types:

File type	Example extensions	How to handle
Tabular text	.csv, .tsv, .txt	Stream-read (chunked) into DuckDB / Polars
Columnar binary	.parquet, .feather, .orc	DuckDB / PyArrow direct read
Excel	.xlsx, .xls	OpenPyXL / DuckDB read
JSON	.json (flat or nested)	Stream parse (ijson) for large; flatten if needed
HTML	.html, .htm	Scrape tables or specific selectors
Compressed archives	.zip, .tar.gz	Extract on the fly; avoid unpacking > needed files
Images (rare for analysis)	.png, .jpg	Usually for OCR or image-feature extraction
Small scripts	.py, .sql	If given as preprocessing scripts (run with sandbox)

