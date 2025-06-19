
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


class Parser:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def normalize_url(self, url):
        try:
            uri = rfc3986.uri_reference(url).normalize()
            return uri.unsplit()
        except Exception as e:
            self.logger.error("Error normalizing %s: %s", url, e)
            return url

    def check_spider_traps(self, url):
        MAX_URL_LENGTH = 200
        MAX_PATH_DEPTH = 2
        trap_pattern = re.compile(r"(calendar|sessionid|track|ref|sort|date=|page=\d{3,})", re.IGNORECASE)

        link = urlparse(url)

        if len(url) > MAX_URL_LENGTH:
            return False
        if link.path.count('/') > MAX_PATH_DEPTH:
            return False
        if trap_pattern.search(url):
            return False
        logging.info(f"{url} è sicuro")
        return True

    async def fetch_sitemap(self, session, sitemap_url):
        try:
            async with session.get(sitemap_url, timeout=10) as response:
                if response.status != 200:
                    self.logger.warning(f"Failed to fetch {sitemap_url} (Status {response.status})")
                    return []

                content_type = response.headers.get("Content-Type", "").lower()
                content = await response.text()

                # XML Sitemap
                if "xml" in content_type:
                    try:
                        root = ET.fromstring(content.encode())
                    except ET.ParseError as e:
                        self.logger.error(f"XML parse error in {sitemap_url}: {e}")
                        return []

                    namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

                    if root.tag.endswith('index'):
                        sub_sitemaps = [loc.text for loc in root.findall('ns:sitemap/ns:loc', namespace)]
                        results = []
                        for sub_url in sub_sitemaps:
                            results.extend(await self.fetch_sitemap(session, sub_url))
                        return results

                    elif root.tag.endswith('urlset'):
                        return [{
                            'url': url.findtext('ns:loc', default='', namespaces=namespace),
                            'priority': url.findtext('ns:priority', default=None, namespaces=namespace),
                            'update': url.findtext('ns:changefreq', default=None, namespaces=namespace)
                        } for url in root.findall('ns:url', namespace)]

                # HTML Sitemap
                elif "html" in content_type:
                    soup = BeautifulSoup(content, "html.parser")
                    return [{
                        'url': self.normalize_url(a['href']),
                        'priority': None,
                        'update': None
                    } for a in soup.find_all('a', href=True)]

                else:
                    self.logger.warning(f"Unsupported format: {sitemap_url} ({content_type})")
                    return []

        except aiohttp.ClientError as e:
            self.logger.error(f"Error fetching {sitemap_url}: {e}")
            return []

    async def parse_sitemap(self, sitemap_list):
        if not sitemap_list:
            self.logger.warning("No sitemap available.")
            return pd.DataFrame(columns=['url', 'priority', 'update'])

        all_entries = []
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_sitemap(session, url) for url in sitemap_list if url]
            results = await asyncio.gather(*tasks)

            for entries in results:
                for entry in entries:
                    url = self.normalize_url(entry['url'])
                    if self.check_spider_traps(url):
                        entry['url'] = url
                        all_entries.append(entry)

        print(f"sitemap urls{all_entries}")
        return pd.DataFrame(all_entries, columns=['url', 'priority', 'update']).drop_duplicates()

    def extract_links(self, html, base_url, sitemaps_urls=None, useragent=None):
     """
     Estrae URL da href/src in html, normalizza e converte in assoluti rispetto a base_url,
     filtra contro spider traps e path disallow del useragent.
     Torna la lista unita di URL vecchi + nuovi, senza duplicati.
     """
     if sitemaps_urls is None:
        sitemaps_urls = []

     soup = BeautifulSoup(html, "html.parser")
     raw_urls = []

     # Estrai href e src
     for tag in soup.find_all(href=True):
        raw_urls.append(tag['href'].strip())
     for tag in soup.find_all(src=True):
        raw_urls.append(tag['src'].strip())

     # Converti in URL assoluti
     absolute_urls = [urljoin(base_url, url) for url in raw_urls]

     # Normalizza
     normalized_urls = [self.normalize_url(url) for url in absolute_urls]

     # Filtra per robots.txt disallow
     if useragent and hasattr(useragent, 'path_disallow'):
        disallowed = useragent.path_disallow
        normalized_urls = [
            url for url in normalized_urls
            if all(path not in url for path in disallowed)
        ]

     # Filtra spider traps
     filtered_urls = [url for url in normalized_urls if self.check_spider_traps(url)]

     # Unisci con sitemap_urls e rimuovi duplicati
     all_urls = set(sitemaps_urls).union(filtered_urls)

     return list(all_urls)

    def parse_page_tags_all(self, html, tags_type=None):
        if tags_type is None:
            tags_type = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'span', 'a']

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(['script', 'style', 'footer', 'nav', 'noscript', 'header', 'form', 'aside']):
            tag.decompose()

        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        tags = soup.find_all(tags_type)
        print(f"sono state trovate {len(tags)} parole")
        return [tag.get_text(separator=' ', strip=True) for tag in tags]
