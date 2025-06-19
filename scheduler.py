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
from storage import Storage  

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
    def __init__(self, max_concurrency, num_spiders):
        self.frontier = asyncio.Queue() # URLs to crawl
        self.seen = set()   # tracks seen URLs
        self.visited = set()    # tracks visited URLs 
        self.semaphore = asyncio.Semaphore(max_concurrency) # limits max parallel fetches
        self.host_locks = defaultdict(asyncio.Lock) #{}  # ensures one fetch per host at a time WILL REUSE SAME LOCK?
        self.retries = defaultdict(int) # dictionary keeps count of how many times retried each URL

        self.max_concurrency = max_concurrency
        self.num_spiders = num_spiders

        self.fetcher = Fetcher()
        self.parser = Parser()
        self.storage = Storage ()
 
    async def add_url(self, url):
        """
        Add a new URL to the frontier if it hasn't been seen.
        """
        if url not in self.seen:
            self.seen.add(url)
            await self.frontier.put(url)
            logger.info(f"Added URL to frontier: {url}")

    async def seed_urls(self, urls):
        """
        Seeds the initial URL, or list of URLs, into the queue.
        """
        for url in urls:
            await self.add_url(url)   # passing it to add_url for check

    def get_hostname(self, url):
        """
        Extract the hostname from a URL.
        """
        return urlparse(url).netloc
    
    async def get_url(self):
        """
        Get the next URL from the frontier.
        Wait asynchronously for a URL to be available in the queue.
        """
        url = await self.frontier.get()
        logger.info(f"Got URL from frontier: {url}")
        return url

    def task_done(self):
        """
        Mark the current task as completed in the frontier.
        """
        self.frontier.task_done()
        logger.info(f"Task done. Frontier size: {self.frontier.qsize()}")

    async def handle_fetch_failure(self, url, exception):
        """
        Handles a failed fetch attempt: retries up to 2 times.
        """
        logger.error(f"Fetch failed for {url}: {exception}")
        self.retries[url] += 1

        if self.retries[url] <= 2:
            logger.info(f"Re-queuing {url} (attempt {self.retries[url]})")
            await self.frontier.put(url)
        else:
            logger.info(f"Giving up on {url} after {self.retries[url]} attempts.")

    async def fetch_url(self, url):
        """
        Fetches a URL with concurrency and politeness constraints.
        Handles retry on failure.
        """
        hostname = self.get_hostname(url)

        # Check freshness: skip if still fresh
        if not self.storage.needs_refresh(url):
            logger.info(f"[SKIP] {url} is fresh, skipping fetch.")
            return None

        # enforce both global fetch concurrency and per-host politeness
        async with self.semaphore:
            async with self.host_locks[hostname]:
                logger.info(f"Fetching {url}")
                start_time = time.perf_counter()  # want to measure time taken for each request
                try:
                    response = await self.fetcher.fetch(url)
                    duration = time.perf_counter() - start_time
                    logger.info(f"Fetched {url} in {duration:.2f}s")
                    return response
                except aiohttp.ClientError as e:
                    duration = time.perf_counter() - start_time
                    logger.error(f"Fetch failed for {url} after {duration:.2f}s: {e}")
                    await self.handle_fetch_failure(url, e)
                    return None
                except aiohttp.client_exceptions.InvalidURL as e:
                    logger.error(f"Invalid URL for {url}: {e}")
                    await self.handle_fetch_failure(url, e)
                    return None
                except Exception as e:
                    duration = time.perf_counter() - start_time
                    logger.error(f"Fetch failed for {url} after {duration:.2f}s: {e}")
                    await self.handle_fetch_failure(url, e)
                    return None

    async def process_response(self, url, response):
        """
        Handles a successful fetch: parses, stores, and queues new URLs.
        """
        if not response:
            logger.warning(f"Empty response for {url}")
            return

        try:
            content, final_url, status = response
        except Exception as e:
            logger.error(f"Unexpected response format from {url}: {e}")
            return

        # Skip non-successful responses or empty content
        if status != 200 or not content:
            logger.info(f"Skipping {url} due to status {status} or empty content.")
            return
        
        # Check for near-duplicate
        is_dup, dup_url = self.storage.is_near_duplicate(content)
        if is_dup:
            logger.info(f"[SKIP] {url} is a near-duplicate of {dup_url}, skipping save.")
            return

        try:
            await self.storage.save_page(final_url, content)
            logger.info(f"Saved {final_url}")
        except Exception as e:
            logger.error(f"Failed to save {final_url}: {e}")
            return

        # mark as successfully visited only after processing
        self.visited.add(final_url)

        try:
            # Extract and enqueue links found in the page
            links = self.parser.extract_links(content, final_url)
            for link in links:
                await self.add_url(link)
        except Exception as e:
            logger.error(f"Failed to parse links from {final_url}: {e}")

    async def spider(self):
        """
        Spider that pulls a URL, fetches it, parses it, and queues new links.
        """
        while True:
            url = await self.get_url()
            response = await self.fetch_url(url)
            await self.process_response(url, response)
            self.task_done()

    async def run(self, seeds=None):
        """
        Starts the crawling loop with multiple concurrent spiders.
        Creates num_spiders number of tasks.
        Each one runs self.worker(fetcher, parser)
        Uses asyncio.create_task() to run them concurrently
        """
        if not seeds:
            seeds = ["https://example.com"]

        await self.storage.async_init()

        async with self.fetcher:  # manages aiohttp session
            await self.seed_urls(seeds)
            spiders = [asyncio.create_task(self.spider()) for _ in range(self.num_spiders)]

            await self.frontier.join()  # wait for all items in queue to be fully processed
            for s in spiders:
                s.cancel()  # cancel all spiders after done so it doesn't run forever

            await asyncio.gather(*spiders, return_exceptions=True)

            await self.storage.commit()

            logger.info("Crawling finished.")
            logger.info(f"Total seen URLs: {len(self.seen)}")
            logger.info(f"Total successfully visited URLs: {len(self.visited)}")