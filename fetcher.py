import aiohttp
import asyncio
import logging
import time
from urllib.parse import urlparse
import urllib.robotparser
import rfc3986

# questo funziona
class UserAgentPolicy:
    def __init__(self, name, header):
        self.name = name
        self.header = header
        self.base_url = None
        self.path_disallow = None
        self.crawl_delay = None
        self.request_rate = None
        self.last_access = 1e9
        self.visited = set()
        self.lock = asyncio.Lock()


class Fetcher:
    def __init__(self, session, useragent_name="Googlebot"):
        self.session = session
        self.useragents = {
            "Googlebot": UserAgentPolicy("Googlebot", 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'),
            "Bingbot": UserAgentPolicy("Bingbot", 'Mozilla/5.0 (compatible; Bingbot/2.0; +http://www.bing.com/bingbot.htm)'),
            "*": UserAgentPolicy("*", 'Mozilla/5.0 (compatible; MyBot/1.0; +http://example.com/bot)'),
        }
        self.default_agent = self.useragents.get(useragent_name, self.useragents["Googlebot"])


    async def check_robots(self, url, useragent=None):
        if not useragent:
            useragent = self.default_agent
        useragent.base_url = url
        new_url = url.rstrip("/") + "/robots.txt"

        rfl = urllib.robotparser.RobotFileParser()
        try:
            async with self.session.get(new_url, timeout=10) as response:
                if response.status == 200:
                    robots_txt = await response.text()
                    rfl.parse(robots_txt.splitlines())
                    logging.info(f"Fetched robots.txt: {new_url}")
                else:
                    logging.warning(f"Failed to fetch robots.txt: HTTP {response.status}")
                    return None
        except aiohttp.ClientError as e:
            logging.error(f"Error fetching robots.txt: {e}")
            return None

        useragent.path_disallow = rfl.parse("Disallow")
        useragent.crawl_delay = rfl.crawl_delay(useragent.name)
        useragent.request_rate = rfl.request_rate(useragent.name)

        return rfl.site_maps()

    async def check_time(self, useragent):
        async with useragent.lock:
            if useragent.request_rate and useragent.crawl_delay:
                request_delay = (useragent.request_rate[1] * 1e9) / useragent.request_rate[0]
                crawl_delay = useragent.crawl_delay * 1e9
                delay = max(request_delay, crawl_delay)
            elif useragent.crawl_delay:
                delay = useragent.crawl_delay * 1e9
            elif useragent.request_rate:
                delay = (useragent.request_rate[1] * 1e9) / useragent.request_rate[0]
            else:
                delay = 1.5 * 1e9  # default delay
            logging.info(f"Delay: {delay / 1e9} seconds")
            now = time.monotonic_ns()
            wait = max(0, (useragent.last_access + delay) - now)
            if wait > 0:
                await asyncio.sleep(wait / 1e9)
            useragent.last_access = time.monotonic_ns()
            logging.info(f"{useragent.name} Last access: {useragent.last_access / 1e9} seconds")

    async def fetch(self, url, useragent=None):
      if not useragent:
          useragent = self.default_agent

      if url in useragent.visited:
          logging.warning(f"[{useragent.name}] Page already visited: {url}")
          return None

      await self.check_time(useragent)

      headers = {'User-Agent': useragent.header}

      start_time = time.perf_counter()
      try:
          async with self.session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=20)) as response:
              duration = time.perf_counter() - start_time

              if response.status == 200:
                  content_type = response.headers.get('Content-Type', '').lower()
                  if 'text/html' in content_type:
                      html = await response.text()
                      logging.info(f"[{useragent.name}] Fetched HTML page: {url} in {duration:.2f}s")
                      useragent.visited.add(url)
                      return (html, str(response.url), response.status)
                  else:
                      logging.warning(f"[{useragent.name}] Skipping non-HTML content: {url} (Content-Type: {content_type})")
                      return None
              else:
                  logging.warning(f"[{useragent.name}] Failed to fetch {url}: HTTP {response.status} in {duration:.2f}s")
                  return None

      except asyncio.TimeoutError:
          duration = time.perf_counter() - start_time
          logging.error(f"[{useragent.name}] Timeout after {duration:.2f}s fetching {url}")
          return None
      except aiohttp.ClientError as e:
          duration = time.perf_counter() - start_time
          logging.error(f"[{useragent.name}] Client error fetching {url} after {duration:.2f}s: {e}")
          return None
      except Exception as e:
          duration = time.perf_counter() - start_time
          logging.error(f"[{useragent.name}] Unexpected error fetching {url} after {duration:.2f}s: {e}")
          return None



useragent_dict = {
    "Googlebot": UserAgentPolicy("Googlebot",'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'),

    "Bingbot": UserAgentPolicy("Bingbot",'Mozilla/5.0 (compatible; Bingbot/2.0; +http://www.bing.com/bingbot.htm)'),

    "Slurp": UserAgentPolicy("Slurp",'Mozilla/5.0 (compatible; Yahoo! Slurp; http://help.yahoo.com/help/us/ysearch/slurp)'),

    "DuckDuckbot": UserAgentPolicy("DuckDuckbot",'DuckDuckBot/1.0; (+http://duckduckgo.com/duckduckbot.html)'),

    "Yandex": UserAgentPolicy("Yandex", 'Mozilla/5.0 (compatible; YandexBot/3.0; +http://yandex.com/bots)'),

    "*": UserAgentPolicy("*",'Mozilla/5.0 (compatible; MyBot/1.0; +http://example.com/bot)')
}