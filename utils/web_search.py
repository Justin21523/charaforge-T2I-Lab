# backend/utils/web_search.py
import requests
from typing import List, Dict, Any


def search_web(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Mock web search function (replace with real implementation)

    Args:
        query: Search query
        max_results: Maximum number of results to return

    Returns:
        List of search results with title, url, snippet
    """
    # Mock implementation - replace with real web search API
    mock_results = [
        {
            "title": f"Result for '{query}' - Example Site 1",
            "url": f"https://example.com/search?q={query.replace(' ', '+')}&result=1",
            "snippet": f"This is a mock search result for the query '{query}'. This would normally come from a real search engine API.",
        },
        {
            "title": f"Information about {query} - Wikipedia",
            "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
            "snippet": f"Wikipedia article about {query}. Contains comprehensive information on the topic.",
        },
        {
            "title": f"{query} - Latest News",
            "url": f"https://news.example.com/{query.replace(' ', '-')}",
            "snippet": f"Latest news and updates about {query}. Breaking news and current events.",
        },
    ]

    return mock_results[:max_results]


def fetch_url_content(url: str, max_length: int = 1000) -> str:
    """
    Fetch content from a URL (simplified implementation)

    Args:
        url: URL to fetch
        max_length: Maximum content length to return

    Returns:
        Content from the URL or error message
    """
    try:
        # Basic safety check
        if not url.startswith(("http://", "https://")):
            return "Error: Invalid URL protocol"

        response = requests.get(
            url, timeout=10, headers={"User-Agent": "CharaForge-Agent/1.0"}
        )
        response.raise_for_status()

        content = response.text[:max_length]
        if len(response.text) > max_length:
            content += "... (content truncated)"

        return content

    except requests.exceptions.Timeout:
        return "Error: Request timeout"
    except requests.exceptions.RequestException as e:
        return f"Error: Failed to fetch URL - {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"
