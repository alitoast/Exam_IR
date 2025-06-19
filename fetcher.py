import aiohttp
import asyncio
import logging
import time
from urllib.parse import urlparse
import urllib.robotparser
import rfc3986

logger = logging.getLogger(__name__)

# User agent headers
USER_AGENTS = {
    "Googlebot": 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
    "Bingbot": 'Mozilla/5.0 (compatible; Bingbot/2.0; +http://www.bing.com/bingbot.htm)',
    "Slurp": 'Mozilla/5.0 (compatible; Yahoo! Slurp; http://help.yahoo.com/help/us/ysearch/slurp)',
    "DuckDuckbot": 'DuckDuckBot/1.0; (+http://duckduckgo.com/duckduckbot.html)',
    "Yandex": 'Mozilla/5.0 (compatible; YandexBot/3.0; +http://yandex.com/bots)',
    "*": 'Mozilla/5.0 (compatible; MyBot/1.0; +http://example.com/bot)'
}

class Fetcher:
    def __init__(self, agent_name="Googlebot"):
        self.agent_name = agent_name
        self.user_agent = USER_AGENTS.get(agent_name, USER_AGENTS["*"])
        self.visited = set()
        self.robots = {}
        self.crawl_delays = {}
        self.last_access_times = {}
        self.disallowed_paths = {}
        self.sitemaps = {}
        self.session = None

    async def start(self):
        self.session = aiohttp.ClientSession()
        return self

    async def finish(self, *args):
        await self.session.close()

    def normalize_url(self, url):
        """Normalize URL using rfc3986."""
        ref = rfc3986.uri_reference(url).normalize()
        return ref.unsplit()

    def get_hostname(self, url):
        return urlparse(url).netloc

    def init_robot_parser(self, base_url):
        if base_url in self.robots:
            return self.robots[base_url]

        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(f"{base_url}/robots.txt")
        try:
            rp.read()
            self.robots[base_url] = rp
            self.disallowed_paths[base_url] = rp.parse("Disallow")
            self.sitemaps[base_url] = rp.site_maps()
        except Exception as e:
            logger.warning(f"Could not read robots.txt for {base_url}: {e}")
            self.robots[base_url] = None
        return self.robots[base_url]

    def is_allowed(self, url):
        """Check robots.txt for access permission and disallowed paths."""
        base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        rp = self.init_robot_parser(base_url)

        if not rp:
            return True  # fail open if robots.txt can't be read

        if not rp.can_fetch(self.agent_name, url):
            return False

        disallowed_paths = self.disallowed_paths.get(base_url, [])
        for path in disallowed_paths:
            if url.startswith(path):
                return False

        return True

    def get_crawl_delay(self, url):
        """Get crawl delay (in seconds) from robots.txt, default to 1.5s."""
        hostname = self.get_hostname(url)
        if hostname in self.crawl_delays:
            return self.crawl_delays[hostname]

        rp = self.init_robot_parser(f"{urlparse(url).scheme}://{hostname}")
        delay = rp.crawl_delay(self.agent_name) if rp else None
        self.crawl_delays[hostname] = delay if delay is not None else 1.5
        return self.crawl_delays[hostname]

    async def wait_if_needed(self, url):
        """Respect crawl delay based on host and last access time."""
        hostname = self.get_hostname(url)
        delay = self.get_crawl_delay(url)
        now = time.monotonic()

        last_access = self.last_access_times.get(hostname, 0)
        wait_time = max(0, delay - (now - last_access))

        if wait_time > 0:
            await asyncio.sleep(wait_time)

        self.last_access_times[hostname] = time.monotonic()

    async def fetch(self, url):
        """Main async fetch method that respects robots.txt and rate limits."""
        url = self.normalize_url(url)

        logger.debug(f"Fetching URL: {url}")

        if url in self.visited:
            logger.info(f"Already visited: {url}")
            return "", url, None

        if not self.is_allowed(url):
            logger.warning(f"Blocked by robots.txt: {url}")
            return "", url, None

        await self.wait_if_needed(url)

        headers = {'User-Agent': self.user_agent}
        try:
            async with self.session.get(url, headers=headers, timeout=10) as response:
                status = response.status
                final_url = str(response.url)
                
                logger.debug(f"Response status for {url}: {status}")

                if status == 200:
                    html = await response.text()
                    logger.info(f"Fetched {final_url} (status {status})")
                    self.visited.add(url)
                    return (html, final_url, status)
                else:
                    logger.warning(f"Failed to fetch {final_url} (status {status})")
                    return ("", final_url, status)
        except aiohttp.ClientError as e:
            logger.error(f"Client error while fetching {url}: {e}")
            return "", url, None
        except Exception as e:
            logger.error(f"Unexpected error while fetching {url}: {e}")
            return (None, url, None)