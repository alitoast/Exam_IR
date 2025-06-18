
"""
Web Parser Module
=================

Description:
------------
This module provides asynchronous tools for analyzing and extracting structured information 
from web pages and sitemaps. Designed for integration with web crawlers or scrapers, it helps 
normalize URLs, detect spider traps, parse sitemap files, and extract textual and hyperlink data 
from HTML content.

Key Features:
-------------
- RFC 3986-compliant URL normalization
- Heuristic spider trap detection (e.g., session IDs, calendar loops, deep paths)
- Asynchronous sitemap parsing with support for:
  - XML sitemaps (including nested <sitemapindex>)
  - HTML sitemaps as fallback
- Extraction and filtering of links from HTML (`href`, `src`)
- Extraction of visible text content from relevant DOM elements

Core Functions:
---------------

1. **normalize_url(url)**:
   - Cleans and standardizes a URL using the `rfc3986` library
   - Returns a normalized URL string (or the original if normalization fails)

2. **check_spider_traps(url)**:
   - Applies heuristics to identify suspicious URLs (e.g., excessive length, deep paths, tracking params)
   - Returns `True` if the URL appears safe to crawl; otherwise, `False`

3. **async fetch_sitemap(session, sitemap_url)**:
   - Asynchronously downloads and parses a single sitemap (XML or HTML)
   - Supports recursive parsing of nested sitemap indexes
   - Returns a list of dictionary entries with keys: `url`, `priority`, and `update`

4. **async parse_sitemap(sitemap_list)**:
   - Parses multiple sitemaps concurrently using aiohttp and asyncio
   - Normalizes and filters the collected URLs
   - Returns a `pandas.DataFrame` with columns: `url`, `priority`, and `update`

5. **parse_page_url(html, sitemaps_urls, useragent)**:
   - Parses a web page's HTML content and extracts candidate URLs from `href` and `src` attributes
   - Filters out:
     - Duplicates
     - Disallowed paths (via `robots.txt` user-agent rules)
     - Known sitemap URLs
     - Potential spider traps
   - Returns a list of new URLs for crawling

6. **parse_page_tags_all(html, tags_type=['p', 'article', 'section', 'li', 'h1', 'h2', 'h3'])**:
   - Extracts visible textual content from HTML by selecting specified tag types
   - Returns an ordered list of clean text segments for further NLP or indexing

Dependencies:
-------------
- `aiohttp`             : Asynchronous HTTP client for fetching sitemaps
- `asyncio`             : Event loop for concurrent tasks
- `re`                  : Regular expressions for trap detection
- `urllib.parse`        : URL parsing and decomposition
- `xml.etree.ElementTree`: Lightweight XML parsing
- `bs4` (BeautifulSoup) : HTML parsing for content and link extraction
- `rfc3986`             : Standards-compliant URL normalization
- `pandas`              : DataFrame output for sitemap URLs

Use Cases:
----------
- Asynchronous sitemap processing for large-scale web crawling
- Prevent infinite crawling or trap URLs via filtering
- Structured extraction of content for indexing, search, or NLP
- Integration with scraper pipelines, link schedulers, or site auditing tools

"""


# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


def normalize_url(url):

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

def check_spider_traps(url):

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


async def fetch_sitemap(session, sitemap_url):
    '''
    Input:
    session : aiohttp.ClientSession
        An existing aiohttp session used to make the HTTP request.

    sitemap_url : str
        The URL pointing to a sitemap (typically found in robots.txt or known ahead of time).

    Output:
    list of dict
        A list of dictionaries containing:
            - 'url': the page URL
            - 'priority': optional priority value (from XML)
            - 'update': optional change frequency (from XML)
        If the sitemap is invalid or inaccessible, returns an empty list.

    Description:
    Asynchronously fetches and parses a single sitemap (either XML or HTML) and returns a list of URLs of sitemaps
    This function supports:
        - XML sitemaps (including recursive sitemap index support)
        - HTML sitemaps (basic <a> link extraction)
    It validates HTTP status, parses content based on MIME type, and returns structured data
    ready for further processing or filtering.
    '''

    try:
        # Perform an asynchronous HTTP GET request for the sitemap URL
        async with session.get(sitemap_url, timeout=10) as response:
            
            # Handle HTTP failure (non-200 status)
            if response.status != 200:
                logger.warning(f"Failed to fetch {sitemap_url} (Status {response.status})")
                return []

            # Extract MIME type and content
            content_type = response.headers.get("Content-Type", "").lower()
            content = await response.text()

            # XML Sitemap
            if "xml" in content_type:
                try:
                    # Parse the XML content into an ElementTree structure
                    root = ET.fromstring(content.encode())
                except ET.ParseError as e:
                    logger.error(f"XML parse error in {sitemap_url}: {e}")
                    return []

                # Define the XML namespace used by standard sitemaps
                namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

                # Handle <sitemapindex> files (recursive fetching of sub-sitemaps)
                if root.tag.endswith('index'):
                    sub_sitemaps = [loc.text for loc in root.findall('ns:sitemap/ns:loc', namespace)]
                    results = []
                    for sub_url in sub_sitemaps:
                        # Recursively fetch and parse each child sitemap
                        results.extend(await fetch_sitemap(session, sub_url))
                    return results

                # Handle regular <urlset> sitemap with actual page URLs
                elif root.tag.endswith('urlset'):
                    return [{
                        'url': url.findtext('ns:loc', default='', namespaces=namespace),
                        'priority': url.findtext('ns:priority', default=None, namespaces=namespace),
                        'update': url.findtext('ns:changefreq', default=None, namespaces=namespace)
                    } for url in root.findall('ns:url', namespace)]

            #HTML Sitemap 
           
            elif "html" in content_type:
                soup = BeautifulSoup(content, "html.parser")
                return [{
                    'url': normalize_url(a['href']),  # Normalize the extracted link
                    'priority': None,                 # HTML sitemaps usually don't include priority
                    'update': None
                } for a in soup.find_all('a', href=True)]

            # Unsupported format
            else:
                logger.warning(f"Unsupported format: {sitemap_url} ({content_type})")
                return []

    # Handle network or HTTP client exceptions
    except aiohttp.ClientError as e:
        logger.error(f"Error fetching {sitemap_url}: {e}")
        return []


async def parse_sitemap(sitemap_list):
    '''
    Input:
        sitemap_list (list): A list of sitemap URLs to parse.

    Output:
          Returns a DataFrame containing:
         - URL
         - Priority
         - Update frequency


    Description:
         This function uses aiohttp to fetch sitemaps asynchronously. It supports:
            - XML sitemaps (standard or sitemap index)
            - HTML-based sitemaps as fallback
        It normalizes the URLs, filters out potential spider traps,
        and merges the results into a clean DataFrame for crawling or analysis.
    '''
    
    # If the list is empty, return an empty DataFrame
    if not sitemap_list:
        logger.warning("No sitemap available.")
        return pd.DataFrame(columns=['url', 'priority', 'update'])

    all_entries = []  # Accumulate all parsed and cleaned URL entries here

    # Create an asynchronous HTTP session for efficient reuse of connections
    async with aiohttp.ClientSession() as session:
        
        # Create a list of async tasks for all sitemap URLs to fetch and parse them concurrently
        tasks = [fetch_sitemap(session, sitemap_url) for sitemap_url in sitemap_list if sitemap_url]

        # Run all fetch tasks concurrently and wait for them to complete
        results = await asyncio.gather(*tasks)

        # Process each list of entries returned from individual sitemap URLs
        for entries in results:
            for entry in entries:
                url = normalize_url(entry['url'])  # Normalize the URL to a standard format
                if check_spider_traps(url):        # Filter out known spider traps
                    entry['url'] = url
                    all_entries.append(entry)       # Keep the cleaned and validated entry

    # Convert list of entries into a DataFrame and remove duplicate URLs
    df = pd.DataFrame(all_entries, columns=['url', 'priority', 'update']).drop_duplicates()

    return df



def parse_page_url(html,sitemaps_urls,useragent=default_agent):

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

  # Filter out disallowed URLs based on robots.txt rules
  if useragent.path_disallow != None:
    for url in urls:
      for path in useragent.path_disallow:
        if path in url:
         urls.remove(url)

  # Normalize URLs (e.g., remove fragments, resolve relative paths, etc.)
    urls = [normalize_url(url) for url in urls]

  # Remove duplicate URLs
    urls = list(set(urls))

  # Filter out potential spider traps
    urls = [url for url in urls if check_spider_traps(url)]

  # Filter out URLs that are already in the sitemap
  if sitemaps_urls != None:
    def_urls = sitemaps_urls
    for url in urls:
      if url not in def_urls:
         def_urls.append(url)
  else:
    def_urls = urls

  return def_urls


def parse_page_tags_all(html,tags_type = None):

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


