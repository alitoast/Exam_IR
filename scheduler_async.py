"""
implement async crawler to avoid blocking on requests.

max_concurrent: Limits simultaneous fetches (via semaphore)

num_spiders: Number of async crawling tasks running concurrently

maybe defaultdict for lock and retries ain't a good idea, to see if it's best to 
place general dictionary 
"""
import asyncio
import aiohttp
import time
from urllib.parse import urlparse
from collections import defaultdict
import logging

from fetcher import Fetcher  
from parser import Parser    
from storage_async import Storage  

# logging set up
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", # timestamp, level, message
    datefmt="%H:%M:%S",
    filename="crawler.log",  # log file
    filemode="a"  # append to the log file ('w' to overwrite)
)

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, max_concurrency, num_spiders, fetcher):
        self.running = True
        self.frontier = asyncio.Queue()
        self.seen = set()
        self.visited = set()
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.host_locks = defaultdict(lambda: asyncio.Lock())
        self.retries = defaultdict(int)

        self.max_concurrency = max_concurrency
        self.num_spiders = num_spiders

        self.fetcher = fetcher
        self.parser = Parser()
        self.storage = Storage()

    async def add_url(self, url):
        if url not in self.seen:
            self.seen.add(url)
            await self.frontier.put(url)

    async def seed_urls(self, urls):
        for url in urls:
            await self.add_url(url)

    def get_hostname(self, url):
        return urlparse(url).netloc

    async def get_url(self):
        return await self.frontier.get()

    def task_done(self):
        self.frontier.task_done()

    async def fetch_url(self, url):
        hostname = self.get_hostname(url)

        async with self.semaphore:
            async with self.host_locks[hostname]:
                response = await self.fetcher.fetch(url)
                return response

    async def process_response(self, url, response):
        if not response:
            logger.warning(f"Empty response for {url}")
            return

        try:
            content, final_url, status = response
        except Exception as e:
            logger.error(f"Unexpected response format from {url}: {e}")
            print(f"Unexpected response format from {url}: {e}")
            return

        if status != 200 or not content:
            logger.info(f"Skipping {url} due to status {status} or empty content.")
            print(f"Skipping {url} due to status {status} or empty content.")
            return

        await self.storage.save_page(final_url, content)
        self.visited.add(final_url)

        try:
            links = self.parser.extract_links(content, final_url)
            for link in links:
                await self.add_url(link)
        except Exception as e:
            logger.error(f"Failed to parse links from {final_url}: {e}")

    async def spider(self):
        while self.running:
            try:
                url = await asyncio.wait_for(self.get_url(), timeout=2)
            except asyncio.TimeoutError:
                break
            response = await self.fetch_url(url)
            await self.process_response(url, response)
            self.task_done()

    async def run(self, seeds=None):
        if not seeds:
            seeds = ["https://example.com"]

        async with aiohttp.ClientSession() as session:
            self.fetcher = Fetcher(session)
            await self.seed_urls(seeds)


            unique_hosts = {urlparse(url).netloc for url in seeds}
            for host in unique_hosts:
                site_url = f"https://{host}"
                await self.fetcher.check_robots(site_url)

            self.running = True  # Attiva il flag di esecuzione
            spiders = [asyncio.create_task(self.spider()) for _ in range(self.num_spiders)]

            await self.frontier.join()  # Aspetta che la coda sia vuota
            self.running = False        # Ferma gli spider

            # Attendi che gli spider escano dal loop e terminino
            await asyncio.gather(*spiders, return_exceptions=True)

            logger.info("Crawling finished.")
            print("Crawling finished.")
            logger.info(f"Total seen URLs: {len(self.seen)}")
            print(f"Total seen URLs: {len(self.seen)}")
            logger.info(f"Total successfully visited URLs: {len(self.visited)}")
            print(f"Total successfully visited URLs: {len(self.visited)}")

