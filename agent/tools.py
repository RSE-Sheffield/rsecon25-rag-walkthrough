from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool


@tool
def get_wikipedia_info(query: str) -> dict:
    """
    Get information from Wikipedia for a given query.
    You should use only this tool when you need to find general information about a topic or when all other sources fail to return.

    Args:
        query (str): The search query for Wikipedia.

    Returns:
        dict: JSON response containing Wikipedia information.
    """
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wikipedia.run(query)


TOOLS = [
    get_wikipedia_info,
]
