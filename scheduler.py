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
import utils_async

logger = logging.getLogger(__name__)

class Scheduler:
    def __init__(self, max_concurrency, num_spiders, max_depth=2):
        self.frontier = asyncio.Queue()  # URLs to crawl
        self.seen = set()  # tracks seen URLs
        self.visited = set()  # tracks visited URLs
        self.semaphore = asyncio.Semaphore(max_concurrency)  # limits max parallel fetches
        self.host_locks = {}  # ensures one fetch per host at a time
        self.retries = defaultdict(int)  # dictionary keeps count of how many times retried each URL

        self.max_concurrency = max_concurrency
        self.num_spiders = num_spiders
        self.max_depth = max_depth

        self.fetcher = Fetcher(None)
        self.parser = Parser()
        self.storage = Storage()

    async def add_url(self, url, current_depth):
        """
        Add a new URL to the frontier if it hasn't been seen and within depth limit.
        """
        if url not in self.seen and current_depth < self.max_depth:
            self.seen.add(url)
            await self.frontier.put((current_depth, url))
            logger.debug(f"Added URL to frontier: {url} at depth {current_depth}")
        else:
            logger.debug(f"Skipping URL {url} at depth {current_depth} (max depth reached or already seen)")

    async def seed_urls(self, urls, initial_depth=0):
        """
        Seeds the initial URL, or list of URLs, into the queue.
        """
        for url in urls:
            await self.add_url(url, initial_depth)  # passing it to add_url for check

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
        current_depth, url = await self.frontier.get()
        logger.debug(f"Got URL from frontier: {url} at depth {current_depth}")
        return url, current_depth

    def task_done(self):
        """
        Mark the current task as completed in the frontier.
        """
        self.frontier.task_done()
        logger.debug(f"Task done. Frontier size: {self.frontier.qsize()}")

    async def handle_fetch_failure(self, url, depth, exception):
        """
        Handles a failed fetch attempt: retries up to 2 times.
        """
        logger.error(f"Fetch failed for {url}: {exception}")
        self.retries[url] += 1

        if self.retries[url] <= 2:
            logger.info(f"Re-queuing {url} (attempt {self.retries[url]})")
            await self.frontier.put((depth, url)) # tuples
        else:
            logger.info(f"Giving up on {url} after {self.retries[url]} attempts.")

    async def fetch_url(self, url):
        """
        Fetches a URL with concurrency and politeness constraints.
        Handles retry on failure.
        """
        hostname = self.get_hostname(url)

        if hostname not in self.host_locks:
            self.host_locks[hostname] = asyncio.Lock()
        
        lock = self.host_locks[hostname]

        # check freshness: skip if still fresh
        if not self.storage.needs_refresh(url):
            logger.info(f"Skipping {url} is fresh, skipping fetch.")
            return None

        # enforce both global fetch concurrency and per-host politeness
        async with self.semaphore:
            async with lock[hostname]:
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
                    #await self.handle_fetch_failure(url, e)
                    return None
                except aiohttp.client_exceptions.InvalidURL as e:
                    logger.error(f"Invalid URL for {url}: {e}")
                    #await self.handle_fetch_failure(url, e)
                    return None
                except Exception as e:
                    duration = time.perf_counter() - start_time
                    logger.error(f"Fetch failed for {url} after {duration:.2f}s: {e}")
                    #await self.handle_fetch_failure(url, e)
                    return None

    async def process_response(self, url, response, current_depth):
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

        logger.debug(f"Processing response for {url} (status {status})")

        # skip non-successful responses or empty content
        if status != 200 or not content:
            logger.info(f"Skipping {url} due to status {status} or empty content.")
            return

        # check for near-duplicate
        is_dup, dup_url = self.storage.is_near_duplicate(content)
        if is_dup:
            logger.info(f"Skipping {url} is a near-duplicate of {dup_url}, skipping save.")
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
            # extract and enqueue links found in the page
            links = self.parser.extract_links(content, final_url)
            for link in links:
                await self.add_url(link, current_depth + 1)
        except Exception as e:
            logger.error(f"Failed to parse links from {final_url}: {e}")

    async def spider(self):
        """
        Spider that pulls a URL, fetches it, parses it, and queues new links.
        """
        while self.running:
            try:
                url, current_depth = await asyncio.wait_for(self.get_url(), timeout=5)
            except asyncio.TimeoutError:
                # if no URL for 5 seconds, exit loop
                break

            response = await self.fetch_url(url)
            await self.process_response(url, response, current_depth)
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
        await self.seed_urls(seeds, initial_depth=0)  # add seeds to frontier

        async with aiohttp.ClientSession() as session:
            self.fetcher = Fetcher(session)  # create Fetcher *after* session

            unique_hosts = {urlparse(url).netloc for url in seeds}
            for host in unique_hosts:
                site_url = f"https://{host}"
                await self.fetcher.check_robots(site_url)

            self.running = True  # execution flag
            spiders = [asyncio.create_task(self.spider()) for _ in range(self.num_spiders)]

            await self.frontier.join()  # wait for frontier to be empty
            self.running = False  # stop spiders

            await asyncio.gather(*spiders, return_exceptions=True)

            await self.storage.commit()

            logger.info("Crawling finished.")
            logger.info(f"Total seen URLs: {len(self.seen)}")
            logger.info(f"Total successfully visited URLs: {len(self.visited)}")