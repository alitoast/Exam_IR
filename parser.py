
"""
Web Parser Module
=================

Description:
------------
This module provides asynchronous tools for extracting structured data from web pages and sitemaps. 
Designed for integration into crawlers, scrapers, or indexing pipelines, it supports URL normalization, 
spider trap detection, XML/HTML sitemap parsing, and text/hyperlink extraction from HTML content.

Key Features:
-------------
- RFC 3986-compliant URL normalization
- Heuristic detection of spider traps (e.g., session IDs, calendar loops, deep paths)
- Asynchronous sitemap parsing with support for:
  - XML sitemaps (including nested `<sitemapindex>`)
  - HTML sitemaps as fallback
- HTML link extraction and filtering from `href` and `src` attributes
- Extraction of visible text content from key HTML tags

Main Functions:
---------------

1. **normalize_url(url)**:
   - Normalizes a given URL using the `rfc3986` library
   - Returns a cleaned and consistent URL string, or the original if normalization fails

2. **check_spider_traps(url)**:
   - Applies heuristics to detect suspicious URLs (e.g., long URLs, deep paths, tracking parameters)
   - Returns `True` if the URL is considered safe to crawl; `False` otherwise

3. **async fetch_sitemap(session, sitemap_url)**:
   - Asynchronously downloads and parses a sitemap (XML or HTML format)
   - Supports recursive parsing of `<sitemapindex>` entries
   - Returns a list of extracted URLs

4. **async parse_sitemap(sitemap_list)**:
   - Concurrently parses multiple sitemap URLs using `aiohttp` and `asyncio`
   - Normalizes and filters collected URLs
   - Returns a deduplicated list of valid URLs

5. **extract_links(html, base_url, sitemaps_urls=None, useragent=None)**:
   - Extracts candidate URLs from HTML (`href` and `src` attributes)
   - Converts to absolute URLs, normalizes, and filters:
     - Duplicate URLs
     - Disallowed paths (via `robots.txt`)
     - Already-known sitemap URLs
     - Spider traps
   - Returns a list of new URLs for crawling

6. **parse_page_tags_all(html, tags_type=None)**:
   - Extracts visible text content from specific HTML tags (default: headers, paragraphs, divs, spans, anchors)
   - Returns an ordered list of cleaned text segments for indexing or NLP tasks

Dependencies:
-------------
- `aiohttp`              : Asynchronous HTTP client for fetching sitemaps
- `asyncio`              : Event loop for concurrent task management
- `re`                   : Regular expressions for identifying spider traps
- `urllib.parse`         : Tools for URL manipulation and resolution
- `xml.etree.ElementTree`: Lightweight XML parsing
- `bs4` (BeautifulSoup)  : HTML parsing for links and text content
- `rfc3986`              : Standards-compliant URL normalization
- `pandas` (optional)    : Used for DataFrame manipulation in other parts of the project (not mandatory here)

Use Cases:
----------
- Asynchronous and efficient sitemap crawling
- Filtering out unsafe or trap URLs to prevent infinite loops
- Structured content extraction for indexing, search engines, or NLP applications
- Seamless integration into scraper pipelines, web crawlers, or site auditing tools
"""

import pandas as pd
import re
from urllib.parse import urlparse, urljoin
import urllib.robotparser #per gestire il file robot.txt
import xml.etree.ElementTree as ET  #per gestire i file xml
import time #per gestire il tempo
from bs4 import BeautifulSoup #per gestire il parsing del html (si può usare anche per xml)
import rfc3986 # per la normalizzazione degli urls
import logging
import asyncio
import aiohttp

class Parser:
    def __init__(self):
        """Initializes the parser with a logger instance."""
        self.logger = logging.getLogger(__name__)

    def normalize_url(self, url):
        """
        Normalizes a given URL using RFC3986 standards.

        Args:
            url (str): The URL to normalize.

        Returns:
            str: A clean, normalized URL. If normalization fails, returns the original URL.
        """
        try:
            uri = rfc3986.uri_reference(url).normalize()
            return uri.unsplit()
        except Exception as e:
            self.logger.error("Error normalizing %s: %s", url, e)
            return url

    def check_spider_traps(self, url):
        """
        Detects if a URL is likely to be a spider trap.

        Args:
            url (str): The URL to check.

        Returns:
            bool: False if the URL is considered dangerous or problematic for crawling, True otherwise.
        """
        MAX_URL_LENGTH = 200 # Arbitrary maximum allowed URL length
        MAX_PATH_DEPTH = 4   # Maximum allowed number of slashes in path
        trap_pattern = re.compile(r"(calendar|sessionid|track|ref|sort|date=|page=\d{3,})", re.IGNORECASE) # Pattern matching common signs of spider traps

        link = urlparse(url)

        # Check URL length
        if len(url) > MAX_URL_LENGTH:
            return False
        # Check path depth (number of '/' in path)      
        if link.path.count('/') > MAX_PATH_DEPTH:
            return False
        # Check for suspicious patterns in the URL     
        if trap_pattern.search(url):
            return False
        logging.info(f"{url} è sicuro")
        return True

    async def fetch_sitemap(self, session, sitemap_url):
        """
        Fetches and parses an XML or HTML sitemap from a given URL.

        Args:
            session (aiohttp.ClientSession): Active HTTP session.
            sitemap_url (str): URL of the sitemap.

        Returns:
            List[str]: A list of extracted URLs, or an empty list if parsing fails.
        """
        try:
            # Make a GET request to the sitemap URL
            async with session.get(sitemap_url, timeout=10) as response:
                # If the response is not successful, log a warning and return an empty list
                if response.status != 200:
                    self.logger.warning(f"Failed to fetch {sitemap_url} (Status {response.status})")
                    return []
                
                # Get content type to distinguish between XML and HTML sitemaps
                content_type = response.headers.get("Content-Type", "").lower()
                content = await response.text()

                # XML Sitemap
                if "xml" in content_type:
                    try:
                        root = ET.fromstring(content.encode())
                    except ET.ParseError as e:
                        self.logger.error(f"XML parse error in {sitemap_url}: {e}")
                        return []

                    # Define XML namespace for standard sitemap tags
                    namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

                    # Handle nested sitemap index: fetch sub-sitemaps recursively
                    if root.tag.endswith('index'):
                        sub_sitemaps = [loc.text for loc in root.findall('ns:sitemap/ns:loc', namespace)]
                        results = []
                        for sub_url in sub_sitemaps:
                            results.extend(await self.fetch_sitemap(session, sub_url))
                        return results

                    # Handle regular URL sitemap: extract all URLs
                    elif root.tag.endswith('urlset'):
                        return [
                            url.findtext('ns:loc', default='', namespaces=namespace)
                            for url in root.findall('ns:url', namespace)
                        ]

                # HTML Sitemap
                elif "html" in content_type:
                    soup = BeautifulSoup(content, "html.parser")
                    return [self.normalize_url(a['href']) for a in soup.find_all('a', href=True)]

                # Unsupported Format
                else:
                    self.logger.warning(f"Unsupported format: {sitemap_url} ({content_type})")
                    return []

        # Handle aiohttp network/client errors
        except aiohttp.ClientError as e:
            self.logger.error(f"Error fetching {sitemap_url}: {e}")
            return []

    async def parse_sitemap(self, sitemap_list):
        """
        Concurrently parses a list of sitemap URLs and returns valid, deduplicated URLs.

        Args:
            sitemap_list (List[str]): List of sitemap URLs to process.

        Returns:
            List[str]: Clean, filtered list of crawlable URLs.
        """
        # If no sitemap URLs are provided, log a warning and return an empty list
        if not sitemap_list:
            self.logger.warning("No sitemap available.")
            return []

        all_urls = []
        # Open a single aiohttp session for all HTTP requests
        async with aiohttp.ClientSession() as session:
            # Create a list of tasks for concurrently fetching and parsing each sitemap
            tasks = [self.fetch_sitemap(session, url) for url in sitemap_list if url]

            # Wait for all tasks to complete and gather their results
            results = await asyncio.gather(*tasks)

            # Flatten the results and process each URL
            for urls in results:
                for url in urls:
                    # Normalize the URL
                    url_norm = self.normalize_url(url)
                    # Include only URLs that are not spider traps
                    if self.check_spider_traps(url_norm):
                        all_urls.append(url_norm)

        # Remove duplicates by converting the list to a set and back to a list
        unique_urls = list(set(all_urls))
        # Print the final set of discovered URLs
        print(f"sitemap urls: {unique_urls}")
        return unique_urls

    def extract_links(self, html, base_url, sitemaps_urls=None, useragent=None):
        """
        Extracts and filters links from HTML content.

        Args:
            html (str): The HTML content.
            base_url (str): Base URL to resolve relative paths.
            sitemaps_urls (List[str], optional): URLs from sitemap to include.
            useragent (robotparser.RobotFileParser, optional): Used to apply disallow rules.

        Returns:
            List[str]: A deduplicated list of normalized and safe URLs to crawl.
        """
        # Initialize sitemap URLs if none are provided
        if sitemaps_urls is None:
            sitemaps_urls = []

        soup = BeautifulSoup(html, "html.parser")
        raw_urls = []

        # Extract all 'href' and 'src' attribute values from HTML elements
        for tag in soup.find_all(href=True):
            raw_urls.append(tag['href'].strip())
        for tag in soup.find_all(src=True):
            raw_urls.append(tag['src'].strip())

        # Convert relative URLs into absolute URLs based on the base URL
        absolute_urls = [urljoin(base_url, url) for url in raw_urls]

        # Normalize all URLs to a standard format
        normalized_urls = [self.normalize_url(url) for url in absolute_urls]

        # Filter out URLs disallowed by robots.txt, if provided
        if useragent and hasattr(useragent, 'path_disallow') and useragent.path_disallow:
            disallowed = useragent.path_disallow
            normalized_urls = [
                url for url in normalized_urls
                if all(path not in url for path in disallowed)
            ]

        # Remove suspicious or trap-like URLs
        filtered_urls = [url for url in normalized_urls if self.check_spider_traps(url)]

        # Merge with sitemap URLs and remove duplicates
        all_urls = set(sitemaps_urls).union(filtered_urls)

        return list(all_urls)

    def parse_page_tags_all(self, html, tags_type=None):
        """
        Extracts visible text from specified HTML tags.

        Args:
            html (str): The HTML content.
            tags_type (List[str], optional): List of tag names to extract. Defaults to headings, p, div, etc.

        Returns:
            List[str]: Cleaned text content from each selected tag.
        """
        # Default set of tag types to extract if none provided
        if tags_type is None:
            tags_type = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'span', 'a']

        soup = BeautifulSoup(html, "html.parser")

        # Remove unwanted tags (scripts, styles, layout components)
        for tag in soup(['script', 'style', 'footer', 'nav', 'noscript', 'header', 'form', 'aside']):
            tag.decompose()

        # Remove HTML comments from the content
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Find and extract all relevant tags
        tags = soup.find_all(tags_type)
        print(f"sono state trovate {len(tags)} parole")
        # Clean and return the visible text from each tag
        return [tag.get_text(separator=' ', strip=True) for tag in tags]
