import logging
import re
from urllib.parse import urlparse, urljoin
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup, Comment
import rfc3986
import pandas as pd
import aiohttp
import asyncio

logger = logging.getLogger(__name__)


class Parser:
    def __init__(self, default_agent=None, sitemaps_urls=None):
        """
        Initialize parser with optional default user agent object that contains
        path_disallow rules for filtering URLs, and optional list of known sitemap URLs.
        """
        self.default_agent = default_agent
        self.sitemaps_urls = sitemaps_urls if sitemaps_urls is not None else []

    def normalize_url(self, url):
        try:
            uri = rfc3986.uri_reference(url).normalize()
            return uri.unsplit()
        except Exception as e:
            logger.error("Error normalizing URL %s: %s", url, e)
            return url

    def check_spider_traps(self, url):
        MAX_URL_LENGTH = 200
        MAX_PATH_DEPTH = 2
        trap_pattern = re.compile(r"(calendar|sessionid|track|ref|sort|date=|page=\d{3,})", re.IGNORECASE)

        parsed = urlparse(url)

        if len(url) > MAX_URL_LENGTH:
            return False
        if parsed.path.count('/') > MAX_PATH_DEPTH:
            return False
        if trap_pattern.search(url):
            return False
        return True

    async def fetch_sitemap(self, session, sitemap_url):
        try:
            async with session.get(sitemap_url, timeout=10) as response:
                response.raise_for_status()  # Raise for non-200 HTTP codes

                content_type = response.headers.get("Content-Type", "").lower()
                content = await response.text()

                if "xml" in content_type:
                    try:
                        root = ET.fromstring(content.encode())
                    except ET.ParseError as e:
                        logger.error(f"XML parse error in {sitemap_url}: {e}")
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

                elif "html" in content_type:
                    soup = BeautifulSoup(content, "html.parser")
                    return [{
                        'url': self.normalize_url(a['href']),
                        'priority': None,
                        'update': None
                    } for a in soup.find_all('a', href=True)]

                else:
                    logger.warning(f"Unsupported sitemap format: {sitemap_url} ({content_type})")
                    return []

        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching {sitemap_url}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching {sitemap_url}: {e}")
            return []

    async def parse_sitemap(self, sitemap_list):
        if not sitemap_list:
            logger.warning("No sitemap URLs provided")
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

        df = pd.DataFrame(all_entries, columns=['url', 'priority', 'update']).drop_duplicates()
        return df

    def parse_page_url(self, html, sitemaps_urls=None, useragent=None):
        """
        Extract URLs from HTML, filter by useragent disallowed paths, spider traps,
        and exclude those in sitemaps_urls if given.
        Does NOT modify self.sitemaps_urls; returns the filtered new URLs.
        """
        if sitemaps_urls is None:
            sitemaps_urls = []

        soup = BeautifulSoup(html, 'html.parser')
        urls = []

        for tag in soup.find_all(href=True):
            urls.append(tag['href'])
        for tag in soup.find_all(src=True):
            urls.append(tag['src'])

        path_disallow = []
        if useragent and hasattr(useragent, 'path_disallow'):
            path_disallow = useragent.path_disallow

        if path_disallow:
            urls = [url for url in urls if all(path not in url for path in path_disallow)]

        urls = [self.normalize_url(url) for url in urls]
        urls = list(set(urls))
        urls = [url for url in urls if self.check_spider_traps(url)]

        new_urls = [url for url in urls if url not in sitemaps_urls]

        return new_urls

    def extract_links(self, content, base_url):
        """
        Extract and normalize links from content using base_url.
        """
        soup = BeautifulSoup(content, "html.parser")
        links = []

        for tag in soup.find_all('a', href=True):
            href = tag['href'].strip()
            absolute_url = urljoin(base_url, href)
            normalized_url = self.normalize_url(absolute_url)
            if normalized_url.startswith(('http://', 'https://')) and self.check_spider_traps(normalized_url):
                links.append(normalized_url)

        return list(set(links))

    def parse_page_tags_all(self, html, tags_type=None):
        if tags_type is None:
            tags_type = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'span', 'a']

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(['script', 'style', 'footer', 'nav', 'noscript', 'header', 'form', 'aside']):
            tag.decompose()

        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        tags = soup.find_all(tags_type)

        texts = [tag.get_text(separator=' ', strip=True) for tag in tags]

        return texts
