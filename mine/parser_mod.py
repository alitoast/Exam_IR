"""
parser --> to async, class
"""

import logging
import re
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup, Comment
import rfc3986
import pandas as pd

import aiohttp
import asyncio

from fetcher import Fetcher

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    filename="crawler.log",  # log file
    filemode="a"  # append to the log file ('w' to overwrite)
)

logger = logging.getLogger(__name__)

class Parser:
    def __init__(self, sitemaps_urls=None, useragent=None):
        self.sitemaps_urls = sitemaps_urls or []
        self.useragent = useragent

    def normalize_url(self, url):
        '''
        Input:
            url (str): The URL to be normalized.

        Output:
            str: The normalized URL as a string. If normalization fails, the original URL is returned.

        Description:
            Normalizes the input URL according to the RFC 3986 standard and returns a string representation.
            This function uses the `rfc3986` library to parse and normalize the URL according to the
            URI standard defined in RFC 3986. This includes handling issues such as case normalization,
            removing default ports, sorting query parameters (if applicable), and more.

            If an error occurs during normalization (e.g., invalid input), it catches the exception
            and returns the original URL as a fallback.
        '''

        try:
            uri = rfc3986.uri_reference(url).normalize()
            return uri.unsplit()

        except Exception as e:
            logger.error("Error, impossble to normalize %s: %s", url, e)
            return url

    def check_spider_traps(self, url):
        '''

        Input:
        url(str):The URL to be analyzed.

        Output:
        - Returns False if the URL is suspicious or considered a "spider trap."
        - Returns True if the URL seems safe to crawl.

        Description:
        Checks whether a given URL is potentially harmful or could trap a web crawler
        in infinite loops or unnecessary crawling paths.

        '''

        MAX_URL_LENGTH = 200   # Arbitrary maximum allowed URL length
        MAX_PATH_DEPTH = 6     # Maximum allowed number of slashes in path
        trap_pattern = re.compile(r"(calendar|sessionid|track|ref|sort|date=|page=\d{3,})", re.IGNORECASE)   # Pattern matching common signs of spider traps:
                                                                                                            # calendars, session IDs, tracking params, endless pagination, etc.

        link = urlparse(url)

        # Check URL length
        if len(link) > MAX_URL_LENGTH:
            return False

        # Check path depth (number of '/' in path)
        if link.path.count('/') > MAX_PATH_DEPTH:
            return False

        # Check for suspicious patterns in the URL
        if trap_pattern.search(url):
            return False

        return True

    async def parse_sitemap(self, sitemap_list):
        '''
        Input:
            sitemap_list (list): A list of sitemap URLs to parse.

        Output:
            Returns a DataFrame containing:
            - URL
            - Priority
            - Update frequency

        Description:
            Supports both XML and HTML sitemaps. Handles nested sitemap indexes.
            Returns a DataFrame with the parsed sitemap entries.
        '''
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        all_entries = []

        if not sitemap_list:
            logger.warning("No sitemap available.")
            return pd.DataFrame(columns=['url', 'priority', 'update'])

        async def parse_single_sitemap(sitemap_url):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(sitemap_url, timeout=10) as response:
                        response.raise_for_status()
                        content_type = response.headers.get("Content-Type", "").lower()
                        content = await response.text()

                        # Sitemap XML
                        if "xml" in content_type:
                            try:
                                root = ET.fromstring(content)
                            except ET.ParseError as e:
                                logger.error("Error into XML parsing %s: %s", sitemap_url, e)
                                return []

                            # If it's a sitemap index, recursively parse child sitemaps
                            if root.tag.endswith('index'):
                                sitemap_urls = [loc.text for loc in root.findall('ns:sitemap/ns:loc', namespace)]
                                entries = []
                                for sub_url in sitemap_urls:
                                    entries.extend(await parse_single_sitemap(sub_url))
                                return entries

                            # If it's a standard URL set
                            elif root.tag.endswith('urlset'):
                                return [{
                                    'url': url.findtext('ns:loc', default='', namespaces=namespace),
                                    'priority': url.findtext('ns:priority', default=None, namespaces=namespace),
                                    'update': url.findtext('ns:changefreq', default=None, namespaces=namespace)
                                } for url in root.findall('ns:url', namespace)]

                        # Handle HTML sitemaps (fallback)
                        elif "html" in content_type:
                            soup = BeautifulSoup(content, "html.parser")
                            return [{
                                'url': self.normalize_url(a['href']),
                                'priority': None,
                                'update': None
                            } for a in soup.find_all('a', href=True)]

                        else:
                            logger.warning("Format unknown %s (%s)", sitemap_url, content_type)
                            return []

            except aiohttp.ClientError as e:
                logger.error("Error, impossible to fetch %s: %s", sitemap_url, e)
                return []

        # Process each sitemap URL in the list
        for sitemap_url in sitemap_list:
            if sitemap_url:
                entries = await parse_single_sitemap(sitemap_url)
                for entry in entries:
                    url = self.normalize_url(entry['url'])  # normalize the URL
                    if self.check_spider_traps(url):  # limit spider traps
                        entry['url'] = url

                all_entries.extend(entries)

        # Create a DataFrame and drop duplicates
        df = pd.DataFrame(all_entries, columns=['url', 'priority', 'update']).drop_duplicates()

        return df


    def parse_page_url(self, html):
        '''
        Inputs:
        - html(str): The HTML content of the page.
        - sitemaps_urls(str): A list of URLs already known from sitemaps.
        - useragent(class): A user agent object containing disallowed paths (robots.txt rules).

        Output:
        - A list of new, allowed URLs extracted from the page.

        Description:
        Given an HTML page, this function extracts all URLs and filters them based on:
        - Exclusion of private or disallowed directories (from robots.txt rules)
        - Deduplication with URLs already found in the sitemap
        - Optional normalization and trap filtering
        '''

        # Parse HTML content
        soup = BeautifulSoup(html, 'html.parser')

        urls = []

        # Extract URLs from href attributes (e.g., <a>, <link>, etc.)
        for tag in soup.find_all(href=True):
            urls.append(tag['href'])

        # Extract URLs from src attributes (e.g., <img>, <script>, etc.)
        for tag in soup.find_all(src=True):
            urls.append(tag['src'])

        # Filter out disallowed URLs based on robots.txt rules safely  
        # DO WE NEED THE PATH DISALLOW? either we expose the disallow from fetcher or we take it away fronm here
        if self.useragent.path_disallow:
            urls = [
                url for url in urls
                if all(path not in url for path in self.useragent.path_disallow)
            ]

        # Normalize URLs (e.g., remove fragments, resolve relative paths, etc.)
        urls = [self.normalize_url(url) for url in urls]

        # Remove duplicate URLs
        urls = list(set(urls))

        # Filter out potential spider traps
        urls = [url for url in urls if self.check_spider_traps(url)]

        # Filter out URLs that are already in the 
        if self.sitemaps_urls != None:
            def_urls = self.sitemaps_urls
            for url in urls:
                if url not in def_urls:
                    def_urls.append(url)
        else:
            def_urls = urls

        return def_urls


    def parse_page_tags_all(self, html, tags_type = None):
        '''
        Inputs:
        - html: The HTML content as a string.
        - tags_type: A list of tag names to search for (default includes common content tags).

        Output:
        - A list of text strings extracted from the specified tags, preserving DOM order

        Description:
        Parses the given HTML content and extracts text from specified HTML tags.
        '''
        if tags_type == None:
            tags_type = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'span', 'a']

        # Initialize the HTML parser
        soup = BeautifulSoup(html, "html.parser")

        # Find tag to remove
        for tag in soup(['script','style','footer','nav','noscript','header','form','aside']):
            tag.decompose()

        # Find all comments and removes them
        for comment in soup.find_all(string=lambda text: isinstance(text,comment)):
            comment.extract()

        # Find all tags of the specified types (respects DOM order)
        tags = soup.find_all(tags_type) 

        # Extract clean text from each tag (removing whitespace and combining with spaces)
        texts = [tag.get_text(separator=' ', strip=True) for tag in tags]

        return texts
    
    async def extract_links(self, html): ## might not need to be async
        return self.parse_page_url(html)