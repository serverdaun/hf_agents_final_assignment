import base64
import io
import json
import shutil
import subprocess as sp
import tempfile
import textwrap
from pathlib import Path
from typing import Dict
import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import (
    TAVILY_API_KEY,
    MODEL_NAME,
    MODEL_API_VERSION,
    MODEL_ENDPOINT,
    MODEL_KEY,
)

from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from openai import AzureOpenAI
from faster_whisper import WhisperModel

# =========================================
# Search Tools
# =========================================


@tool
def wiki_search(query: str) -> str:
    """
    Search Wikipedia for a given query, return top 3 results and scrape full content.
    Args:
        query (str): The search query.
    Returns:
        str: Formatted string containing the titles, URLs, content snippets and full webpage content of the top 3 Wikipedia articles.
    """
    docs = WikipediaLoader(query=query, load_max_docs=2).load()

    results = []
    for doc in docs:
        # Get the standard wiki summary
        wiki_summary = f"\nTitle: {doc.metadata.get('title')}\nURL: {doc.metadata.get('source')}\n\n"

        # Scrape and clean the full webpage
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(doc.metadata.get('source'), headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove unwanted elements
            unwanted_elements = [
                '.mw-jump-link', '.mw-editsection', '.reference',  # Wiki specific
                '#mw-navigation', '#mw-head', '#mw-panel',  # Navigation
                '.navbox', '.vertical-navbox', '.sidebar',  # Navigation boxes
                '.noprint', '.printfooter', '.catlinks',  # Printing related
                '#toc', '.toc', '#site-navigation',  # Table of contents
            ]
            for element in soup.select(','.join(unwanted_elements)):
                element.decompose()

            # Get main content area
            content_div = soup.select_one('#mw-content-text')
            if content_div:
                # Remove disambiguation elements if present
                for disambig in content_div.select('.hatnote, .dmbox-disambig'):
                    disambig.decompose()
                full_text = content_div.get_text(separator='\n', strip=True)
            else:
                full_text = soup.get_text(separator='\n', strip=True)

            # Combine wiki summary with cleaned webpage content
            combined_result = f"{wiki_summary}\n### Full Article Content ###\n{full_text}"
            results.append(combined_result)

        except Exception as e:
            print(f"Error scraping Wikipedia page: {e}")
            results.append(wiki_summary)

    # Join all results with clear separators
    formatted_results = "\n\n" + "=" * 20 + "\n\n".join(results)
    return formatted_results


@tool
def tavily_search(query: str) -> str:
    """
    Search Tavily for a given query and return top 3 results.
    Args:
        query (str): The search query.
    Returns:
        str: Formatted string containing the titles, URLs and content of the top 3 Tavily search results.
    """
    results = TavilySearchResults(max_results=5, tavily_api_key=TAVILY_API_KEY).invoke({"query": query})

    # Format the results
    formatted_results = "\n\n\n--------------\n\n\n".join(
        [
            f"*Metadata*:\nTitle: {result.get('title')}\nURL: {result.get('url')}\n\n"
            f"*Content*:\n{result.get('content')}"
            for result in results
        ]
    )

    return formatted_results


@tool
def arxiv_search(query: str) -> str:
    """
    Search Arxiv for a given query and return top 3 results.
    Args:
        query (str): The search query.
    Returns:
        str: Formatted string containing the titles, URLs and content of the top 3 Arxiv search results.
    """
    docs = ArxivLoader(query=query, load_max_docs=5).load()

    # Format the results
    formatted_results = "\n\n\n--------------\n\n\n".join(
        [
            f"*Metadata*:\nTitle: {doc.metadata.get('Title')}\nURL: {doc.metadata.get('Authors')}\n\n"
            f"*Content*:\n{doc.page_content[1000:]}"
            for doc in docs
        ]
    )

    return formatted_results


@tool
def scrape_webpage(url: str) -> str:
    """
    Scrape the main content from a webpage.
    Args:
        url (str): The URL of the webpage to scrape.
    Returns:
        str: The main text content of the webpage.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(['script', 'style']):
            script.decompose()

        # Get text content
        text = soup.get_text(separator='\n', strip=True)
        return text
    except Exception as e:
        return f"Error scraping webpage: {str(e)}"


# =========================================
# Math Tools
# =========================================


@tool
def add(x: float, y: float) -> float:
    """
    Add two numbers.
    Args:
        x (float): First number.
        y (float): Second number.
    Returns:
        float: The sum of x and y.
    """
    return x + y


@tool
def subtract(x: float, y: float) -> float:
    """
    Subtract two numbers.
    Args:
        x (float): First number.
        y (float): Second number.
    Returns:
        float: The difference of x and y.
    """
    return x - y


@tool
def multiply(x: float, y: float) -> float:
    """
    Multiply two numbers.
    Args:
        x (float): First number.
        y (float): Second number.
    Returns:
        float: The product of x and y.
    """
    return x * y


@tool
def divide(x: float, y: float) -> float:
    """
    Divide two numbers.
    Args:
        x (float): First number.
        y (float): Second number.
    Returns:
        float: The quotient of x and y.
    """
    if y == 0:
        raise ValueError("Cannot divide by zero.")
    return x / y


@tool
def power(x: float, y: float) -> float:
    """
    Raise x to the power of y.
    Args:
        x (float): Base number.
        y (float): Exponent.
    Returns:
        float: The result of x raised to the power of y.
    """
    return x ** y


@tool
def sqrt(x: float) -> float:
    """
    Calculate the square root of x.
    Args:
        x (float): The number to find the square root of.
    Returns:
        float: The square root of x.
    """
    if x < 0:
        raise ValueError("Cannot calculate square root of a negative number.")
    return x ** 0.5


@tool
def modulus(x: float, y: float) -> float:
    """
    Calculate the modulus of x and y.
    Args:
        x (float): First number.
        y (float): Second number.
    Returns:
        float: The modulus of x and y.
    """
    return x % y


@tool
def is_commutative(set_elements: list, operation_table: list) -> bool:
    """
    Check if the operation is commutative for the given set and operation table.
    Args:
        set_elements (list): List of elements in the set.
        operation_table (list): 2D list representing the operation table.
    Returns:
        bool: True if commutative, False otherwise.
    """
    n = len(set_elements)
    for i in range(n):
        for j in range(n):
            if operation_table[i][j] != operation_table[j][i]:
                return False
    return True


@tool
def commutativity_counterexample_pairs(set_elements: list, operation_table: list) -> list:
    """
    Return all pairs (as tuples) where commutativity fails: (x, y) such that x*y != y*x.
    Args:
        set_elements (list): List of elements in the set.
        operation_table (list): 2D list representing the operation table.
    Returns:
        list: List of tuples (x, y) where commutativity fails.
    """
    n = len(set_elements)
    pairs = []
    for i in range(n):
        for j in range(n):
            if operation_table[i][j] != operation_table[j][i]:
                pairs.append((set_elements[i], set_elements[j]))
    return pairs


@tool
def commutativity_counterexample_elements(set_elements: list, operation_table: list) -> str:
    """
    Return the set of elements involved in any commutativity counter-example, as a sorted, comma-separated string.
    Args:
        set_elements (list): List of elements in the set.
        operation_table (list): 2D list representing the operation table.
    Returns:
        str: Sorted, comma-separated string of elements involved in any commutativity counter-example.
    """
    involved = set()
    n = len(set_elements)
    for i in range(n):
        for j in range(n):
            if operation_table[i][j] != operation_table[j][i]:
                involved.add(set_elements[i])
                involved.add(set_elements[j])
    return ",".join(sorted(involved))


@tool
def is_associative(set_elements: list, operation_table: list) -> bool:
    """
    Check if the operation is associative for the given set and operation table.
    Args:
        set_elements (list): List of elements in the set.
        operation_table (list): 2D list representing the operation table.
    Returns:
        bool: True if associative, False otherwise.
    """
    n = len(set_elements)
    idx = {e: i for i, e in enumerate(set_elements)}
    for i in range(n):
        for j in range(n):
            for k in range(n):
                a = operation_table[i][j]
                a_idx = idx[a]
                left = operation_table[a_idx][k]
                b = operation_table[j][k]
                b_idx = idx[b]
                right = operation_table[i][b_idx]
                if left != right:
                    return False
    return True


@tool
def find_identity_element(set_elements: list, operation_table: list) -> str:
    """
    Find the identity element in the set, if it exists.
    Args:
        set_elements (list): List of elements in the set.
        operation_table (list): 2D list representing the operation table.
    Returns:
        str: The identity element, or an empty string if none exists.
    """
    n = len(set_elements)
    for i in range(n):
        candidate = set_elements[i]
        is_identity = True
        for j in range(n):
            if operation_table[i][j] != set_elements[j] or operation_table[j][i] != set_elements[j]:
                is_identity = False
                break
        if is_identity:
            return candidate
    return ""


@tool
def find_inverses(set_elements: list, operation_table: list) -> dict:
    """
    For each element, find its inverse with respect to the operation, if it exists.
    Args:
        set_elements (list): List of elements in the set.
        operation_table (list): 2D list representing the operation table.
    Returns:
        dict: Dictionary mapping each element to its inverse (or None if no inverse exists).
    """
    n = len(set_elements)
    identity = find_identity_element(set_elements, operation_table)
    if not identity:
        return {e: None for e in set_elements}
    inverses = {}
    for i in range(n):
        found = None
        for j in range(n):
            if operation_table[i][j] == identity and operation_table[j][i] == identity:
                found = set_elements[j]
                break
        inverses[set_elements[i]] = found
    return inverses


# =========================================
# Image Tools
# =========================================


@tool
def analyze_image(question: str, path: str) -> str:
    """
    Analyze image and answer question regarding it.
    Args:
        question (str): The question to ask about the image.
        path (str): The path to the image file.
    Returns:
        str: The answer to the question about the image.
    """
    client = AzureOpenAI(
        api_version=MODEL_API_VERSION,
        azure_endpoint=MODEL_ENDPOINT,
        api_key=MODEL_KEY,
    )

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise ValueError(f"Image file does not exist: {p}")

    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    with open(p, "rb") as f:
        base64_image = f"data:{mime};base64,{base64.b64encode(f.read()).decode('utf-8')}"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": base64_image}, "detail": "high"}
                ]
            }
        ]
    )

    return response.choices[0].message.content.strip()


# =========================================
# Audio Tools
# =========================================


@tool
def transcribe_audio(path: str) -> str:
    """
    Transcribe audio file and return the text.
    Args:
        path (str): The path to the audio file.
    Returns:
        str: The transcribed text.
    """
    model = WhisperModel(
        model_size_or_path="small",
        device="cpu"
    )

    segments, _ = model.transcribe(
        path,
        vad_filter=True,
        condition_on_previous_text=True,
        beam_size=5
    )
    text = "".join(seg.text for seg in segments).strip()
    return text


# =========================================
# Code Tools
# =========================================

LANG_COMMANDS: Dict[str, callable] = {
    ".py": lambda s, _: [["python3", s.name]],
    ".js": lambda s, _: [["node", s.name]],
    ".ts": lambda s, _: [["deno", "run", "-A", s.name]],
    ".sh": lambda s, _: [["bash", s.name]],
    ".rb": lambda s, _: [["ruby", s.name]],
    ".php": lambda s, _: [["php", s.name]],
    ".go": lambda s, _: [["go", "run", s.name]]
}


@tool
def execute_source_file(path: str, timeout: int = 10) -> str:
    """
    Run the program contained in *path*
    Returns a newline-separated string:
        >>> EXIT_CODE: <int>
        >>> STDOUT: <captured stdout>
        >>> STDERR: <captured stderr>
    Args:
        path (str): The path to the source file.
        timeout (int): The timeout in seconds.
    Returns:
        str: A newline-separated string containing the exit code, stdout, and stderr.
    """
    src = Path(path).expanduser().resolve(strict=True)
    if src.suffix not in LANG_COMMANDS:
        raise ValueError(f"Unsupported file extension: {src.suffix}")

    # Temp work dir for the program
    work = Path(tempfile.mkdtemp(prefix="exec_tool_"))
    shutil.copy(src, work / src.name)

    try:
        full_out, full_err = "", ""

        for cmd in LANG_COMMANDS[src.suffix](src, work):
            proc = sp.run(
                cmd,
                cwd=work,
                text=True,
                capture_output=True,
                timeout=timeout
            )
            full_out += proc.stdout
            full_err += proc.stderr
            exit_code = proc.returncode
            if exit_code != 0:
                break

        return (
            f"EXIT_CODE: {exit_code}\n"
            f"STDOUT: {full_out}\n"
            f"STDERR: {full_err}"
        )

    finally:
        shutil.rmtree(work)


# =========================================
# Tabular data tools
# =========================================

MAX_BYTES_RETURN = 200000


# Helper functions
def _load_table(path: Path, sheet: str) -> pd.DataFrame:
    """
    Load a table from a file.
    Args:
        path (Path): The path to the file.
        sheet (str): The sheet to load.
    Returns:
        pd.DataFrame: The loaded table.
    """
    ext = path.suffix.lower()
    if ext in (".csv", ".tsv"):
        return pd.read_csv(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet)
    if ext in (".parquet"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension: {ext}")


def _safe_truncate(text: str, limit: int = MAX_BYTES_RETURN) -> tuple[str, bool]:
    """
    Truncate text to a given limit.
    Args:
        text (str): The text to truncate.
        limit (int): The limit in bytes.
    Returns:
        tuple[str, bool]: The truncated text and a boolean indicating if truncation occurred.
    """
    utf8 = text.encode("utf-8")
    truncated = len(utf8) > limit
    if truncated:
        utf8 = utf8[:limit]
    return utf8.decode("utf-8", errors="ignore"), truncated


@tool
def interact_tabular(file_path: str, operation: str = "summary", sheet: str = "Sheet1") -> str:
    """
    Interact with a tabular data file, such as a CSV, Excel, or Parquet file.
    Args:
        path (str): The path to the file.
        operation (str): The operation to perform: summary | head [N] | select col1,col2 | filter <expr>
            describe | to_json
        sheet (str): The sheet to load.
    Returns:
        str: The result of the operation.
    """
    path = Path(file_path).expanduser().resolve(strict=True)
    df = _load_table(path, sheet)
    op, *args = operation.lower().split(maxsplit=1)
    if op == "summary":
        result = textwrap.dedent(f"""\
            rows: {len(df)}
            columns: {", ".join(df.columns)}
            dtypes: {df.dtypes.to_string()}
        """)
    elif op == "head":
        n = int(args[0]) if args else 5
        buf = io.StringIO()
        df.head(n).to_json(buf, orient="records", lines=True)
        result = buf.getvalue()
    elif op == "select":
        cols = [c.strip() for c in args[0].split(",")]
        buf = io.StringIO()
        df[cols].to_json(buf, orient="records", lines=True)
        result = buf.getvalue()
    elif op == "filter":
        expr = args[0]
        buf = io.StringIO()
        df.query(expr, engine="python").to_json(buf, orient="records", lines=True)
        result = buf.getvalue()
    elif op == "describe":
        buf = io.StringIO()
        df.describe(include="all").to_json(buf, orient="records", lines=True)
        result = buf.getvalue()
    elif op == "to_json":
        buf = io.StringIO()
        df.to_json(buf, orient="records", lines=True)
        result = buf.getvalue()
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    result, truncated = _safe_truncate(result)

    info = {
        "file": str(path),
        "sheet": sheet,
        "truncated": truncated,
        "rows_returned": result.count("\n") - 1
    }
    return (
        f"OPERATION: {operation}\n"
        f"RESULT:\n{result}\n"
        f"INFO:\n{json.dumps(info, indent=2)}"
    )
