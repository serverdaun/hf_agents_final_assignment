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
