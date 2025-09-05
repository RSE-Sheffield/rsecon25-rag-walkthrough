from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool


@tool
def get_wikipedia_info(query: str) -> dict:
    """
    Fetch information about RSECon from Wikipedia.

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
