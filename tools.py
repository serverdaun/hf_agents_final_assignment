from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from config import TAVILY_API_KEY


#=========================================
# Search Tools
#=========================================
@tool
def wiki_search(query: str) -> str:
    """
    Search Wikipedia for a given query and return top 3 results.
    Args:
        query (str): The search query.
    Returns:
        str: Formatted string containing the titles, URLs and content of the top 3 Wikipedia articles.
    """
    docs = WikipediaLoader(query=query, load_max_docs=3).load()

    # Format the results
    formatted_results = "\n\n\n--------------\n\n\n".join(
        [
            f"*Metadata*:\nTitle: {doc.metadata.get('title')}\nURL: {doc.metadata.get('source')}\n\n"
            f"*Content*:\n{doc.page_content}"
            for doc in docs
        ]
    )

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
    results = TavilySearchResults(max_results=3, tavily_api_key=TAVILY_API_KEY).invoke({"query": query})

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
    docs = ArxivLoader(query=query, load_max_docs=3).load()

    # Format the results
    formatted_results = "\n\n\n--------------\n\n\n".join(
        [
            f"*Metadata*:\nTitle: {doc.metadata.get('Title')}\nURL: {doc.metadata.get('Authors')}\n\n"
            f"*Content*:\n{doc.page_content[1000:]}"
            for doc in docs
        ]
    )

    return formatted_results


#=========================================
# Math Tools
#=========================================
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
