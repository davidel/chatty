import os
import requests
import html as html_parser
import re
from urllib.parse import quote_plus, unquote
from rich.console import Console

from chatty.utils import tool_fetch_url

console = Console()


def tool_search_web(query: str, max_results: int = 10) -> str:
  """Search the web for a query and return formatted results (title, URL, snippet).

  Supports multiple backends based on environment variables:
  1. Tavily: TAVILY_API_KEY
  2. Brave Search: BRAVE_API_KEY
  3. Google Custom Search: GOOGLE_API_KEY and GOOGLE_CSE_ID
  4. Serper: SERPER_API_KEY
  5. SerpApi: SERPAPI_API_KEY
  6. Yahoo HTML Scraper (Fallback when no keys are provided)
  """
  results = []
  backend_used = ""

  tavily_key = os.environ.get("TAVILY_API_KEY")
  brave_key = os.environ.get("BRAVE_API_KEY")
  google_key = os.environ.get("GOOGLE_API_KEY")
  google_cx = os.environ.get("GOOGLE_CSE_ID")
  serper_key = os.environ.get("SERPER_API_KEY")
  serpapi_key = os.environ.get("SERPAPI_API_KEY")

  try:
    if tavily_key:
      backend_used = "Tavily Search API"
      url = "https://api.tavily.com/search"
      payload = {
        "api_key": tavily_key,
        "query": query,
        "max_results": max_results
      }
      r = requests.post(url, json=payload, timeout=10)
      r.raise_for_status()
      data = r.json()
      for item in data.get("results", []):
        results.append({
          "title": item.get("title", ""),
          "url": item.get("url", ""),
          "snippet": item.get("content", "")
        })

    elif brave_key:
      backend_used = "Brave Search API"
      url = "https://api.search.brave.com/res/v1/web/search"
      headers = {
        "Accept": "application/json",
        "X-Subscription-Token": brave_key
      }
      params = {
        "q": query,
        "count": min(max_results, 20)
      }
      r = requests.get(url, headers=headers, params=params, timeout=10)
      r.raise_for_status()
      data = r.json()
      for item in data.get("web", {}).get("results", []):
        results.append({
          "title": item.get("title", ""),
          "url": item.get("url", ""),
          "snippet": item.get("description", "")
        })

    elif google_key and google_cx:
      backend_used = "Google Custom Search API"
      url = "https://www.googleapis.com/customsearch/v1"
      params = {
        "key": google_key,
        "cx": google_cx,
        "q": query,
        "num": min(max_results, 10)  # Google API limit is 10 per request
      }
      r = requests.get(url, params=params, timeout=10)
      r.raise_for_status()
      data = r.json()
      for item in data.get("items", []):
        results.append({
          "title": item.get("title", ""),
          "url": item.get("link", ""),
          "snippet": item.get("snippet", "")
        })

    elif serper_key:
      backend_used = "Serper Google Search API"
      url = "https://google.serper.dev/search"
      headers = {
        "X-API-KEY": serper_key,
        "Content-Type": "application/json"
      }
      payload = {
        "q": query,
        "num": max_results
      }
      r = requests.post(url, json=payload, headers=headers, timeout=10)
      r.raise_for_status()
      data = r.json()
      for item in data.get("organic", []):
        results.append({
          "title": item.get("title", ""),
          "url": item.get("link", ""),
          "snippet": item.get("snippet", "")
        })

    elif serpapi_key:
      backend_used = "SerpApi Google Search API"
      url = "https://serpapi.com/search.json"
      params = {
        "engine": "google",
        "q": query,
        "api_key": serpapi_key,
        "num": max_results
      }
      r = requests.get(url, params=params, timeout=10)
      r.raise_for_status()
      data = r.json()
      for item in data.get("organic_results", []):
        results.append({
          "title": item.get("title", ""),
          "url": item.get("link", ""),
          "snippet": item.get("snippet", "")
        })

    else:
      # Fallback to Yahoo scraper
      backend_used = "Yahoo HTML Scraper (Fallback)"
      url = f"https://search.yahoo.com/search?p={quote_plus(query)}"
      headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0'
      }
      r = requests.get(url, headers=headers, timeout=10)
      r.raise_for_status()
      
      blocks = re.split(r'<div[^>]*class="[^"]*algo-sr[^"]*"', r.text)
      for block in blocks[1:]:
        title_match = re.search(r'<h3[^>]*>.*?<span[^>]*>(.*?)</span>', block, re.DOTALL)
        if not title_match:
          title_match = re.search(r'<h3[^>]*>(.*?)</h3>', block, re.DOTALL)
          
        url_match = re.search(r'href="([^"]+)"', block)
        
        snippet_match = re.search(r'<div[^>]*class="[^"]*compText[^"]*"[^>]*>(.*?)</div>', block, re.DOTALL)
        if not snippet_match:
          snippet_match = re.search(r'<p[^>]*class="[^"]*fc-dustygray[^"]*"[^>]*>(.*?)</p>', block, re.DOTALL)
          
        if title_match and url_match:
          title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()
          title = html_parser.unescape(title)
          
          raw_url = url_match.group(1)
          url_val = raw_url
          if "r.search.yahoo.com" in raw_url and "/RU=" in raw_url:
            ru_match = re.search(r'/RU=([^/]+)/', raw_url)
            if ru_match:
              url_val = unquote(ru_match.group(1))
              
          snippet = ""
          if snippet_match:
            snippet = re.sub(r'<[^>]+>', '', snippet_match.group(1)).strip()
            snippet = html_parser.unescape(snippet)
            
          if title and not title.lower().startswith("ad") and not "related searches" in title.lower():
            results.append({
              "title": title,
              "url": url_val,
              "snippet": snippet
            })
            if len(results) >= max_results:
              break

    if not results:
      return f"[{backend_used}] No results found."
      
    formatted = [f"Search Engine backend: {backend_used}\n"]
    for idx, res in enumerate(results[:max_results], 1):
      formatted.append(f"[{idx}] {res['title']}\n    URL: {res['url']}\n    Snippet: {res['snippet']}")
    return "\n\n".join(formatted)

  except Exception as e:
    return f"Error searching the web ({backend_used}): {str(e)}"
