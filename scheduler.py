"""
Async Scheduler for a web crawler.
Manages URL frontier, fetch concurrency, depth/budget limits, retry logic,
duplicate detection, and delegation to fetcher, parser, and storage subsystems.

Key features:
- Async crawling using asyncio/aiohttp
- Per-host politeness via asyncio.Lock()
- Global concurrency limit via asyncio.Semaphore()
- Retry mechanism
- Depth & per-domain page budget control
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
    def __init__(self, max_concurrency, num_spiders, max_depth, max_pages_per_domain):
        """
        Initializes the crawler scheduler.

        Args:
            max_concurrency (int): Global max concurrent fetches.
            num_spiders (int): Number of concurrent spider tasks.
            max_depth (int): Maximum crawl depth per seed.
            max_pages_per_domain (int): Max pages to crawl per domain.
        """

        self.frontier = asyncio.Queue() # URLs to crawl
        self.seen = set()               # tracks seen URLs
        self.visited = set()            # tracks visited URLs

        # initialize synchronization primitives
        self.semaphore = asyncio.Semaphore(max_concurrency)  # global concurrency limit
        self.host_locks = {}    # per-host locks for politeness

        # initialize tracking dictionaries
        self.retries = defaultdict(int)  # dictionary keeps count of how many times retried each URL
        self.domain_counts = defaultdict(int)   
        
        # store configuration parameters
        self.max_pages_per_domain = max_pages_per_domain  # prevent crawling too many pages from a single domain
        self.max_concurrency = max_concurrency
        self.num_spiders = num_spiders
        self.max_depth = max_depth

        self.fetcher = Fetcher(None)
        self.parser = Parser()
        self.storage = Storage()

    async def add_url(self, url, current_depth):
        """
        Adds a URL to the frontier queue if unseen and within limits.

        Args:
            url (str): The URL to add.
            current_depth (int): Current depth of crawling.
        """
        # url normalization 
        normalized_url = self.parser.normalize_url(url)
        domain = urlparse(normalized_url).netloc

        # check if we've reached max pages for this domain
        if self.domain_counts[domain] >= self.max_pages_per_domain:
            logger.debug(f"Skipping URL {url} (max pages for domain reached)")
            return

        # add to queue only if unseen and depth is within limit
        if normalized_url not in self.seen:
            # increment depth when adding to frontier
            next_depth = current_depth + 1
            if next_depth < self.max_depth:
                self.seen.add(normalized_url)
                await self.frontier.put((next_depth, normalized_url))
                logger.info(f"Added URL to frontier: {normalized_url} at depth {current_depth + 1}")
                self.domain_counts[domain] += 1

                # attach crawl depth to storage metadata if available
                if url in self.storage.pages:
                    self.storage.pages[url]['current_depth'] = next_depth - 1
            else:
                logger.debug(f"Skipping URL {url} (max depth reached)")
        else:
            logger.debug(f"Skipping URL {url} at depth {current_depth} (already seen)")

    async def seed_urls(self, urls, initial_depth=0):
        """
        Seeds initial URLs into the frontier queue.

        Args:
            urls (list): List of seed URLs.
            initial_depth (int): Depth level to start from.
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
        Retrieves the next URL from the frontier queue.

        Returns:
            tuple: (url, depth)
        """
        current_depth, url = await self.frontier.get()
        logger.debug(f"Got URL from frontier: {url} at depth {current_depth}")
        return url, current_depth

    def task_done(self):
        """
        Marks a frontier task as complete.
        """
        self.frontier.task_done()
        logger.debug(f"Task done. Frontier size: {self.frontier.qsize()}")

    async def handle_fetch_failure(self, url, depth, exception):
        """
        Handles fetch failure with up to 2 retries.
        """
        logger.error(f"Fetch failed for {url}: {exception}")
        self.retries[url] += 1

        if self.retries[url] <= 2:
            # re-queue with the same depth after a bit
            await asyncio.sleep(2 ** (self.retries[url]))
            await self.frontier.put((depth, url))
            logger.info(f"Re-queuing {url} (attempt {self.retries[url]})")
        else:
            logger.info(f"Giving up on {url} after {self.retries[url]} attempts.")

    async def fetch_url(self, url, initial_depth):
        """
        Fetches the URL with concurrency and politeness constraints.

        Args:
            url (str): URL to fetch.
            initial_depth (int): Fallback crawl depth.

        Returns:
            tuple | None: (content, final_url, status), or None if failed
        """
        hostname = self.get_hostname(url)

        if hostname not in self.host_locks:
            self.host_locks[hostname] = asyncio.Lock()
        
        lock = self.host_locks[hostname]

        # check freshness: skip if still fresh
        if not self.storage.needs_refresh(url):
            logger.info(f"[SKIP] {url} is fresh, not fetching again.")

            # get the current depth from storage
            page = self.storage.get_page(url)
            current_depth = getattr(page, 'current_depth', initial_depth)
            
            # check and handle outlinks if necessary
            outlinks = self.storage.get_outlinks(url)
            logger.info(f"[OUTLINKS] Checking {len(outlinks)} links from fresh page: {url}")
            
            for link in outlinks:
                if self.storage.needs_refresh(link):
                    next_depth = min(current_depth + 1, self.max_depth)
                    await self.add_url(link, next_depth)
    
            return None  # skip fetch, links already handled

        # acquire both global and per-host locks
        async with self.semaphore:
            async with lock:
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
                    await self.handle_fetch_failure(url, current_depth, e)
                    return None
                except aiohttp.client_exceptions.InvalidURL as e:
                    logger.error(f"Invalid URL for {url}: {e}")
                    return None
                except Exception as e:
                    duration = time.perf_counter() - start_time
                    logger.error(f"Fetch failed for {url} after {duration:.2f}s: {e}")
                    await self.handle_fetch_failure(url, current_depth, e)
                    return None

    async def process_response(self, url, response, current_depth):
        """
        Processes a successful fetch: saves, parses, and queues outlinks.

        Args:
            url (str): Original URL.
            response (tuple): (content, final_url, status)
            current_depth (int): Crawl depth.
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

        self.visited.add(final_url)

        # parse and enqueue outlinks
        try:
            links = self.parser.extract_links(content, final_url,
                                        sitemaps_urls=self.storage.get_outlinks(final_url),
                                        useragent=self.fetcher.default_agent)
            for link in links:
                await self.add_url(link, current_depth + 1)
        except Exception as e:
            logger.error(f"Failed to parse links from {final_url}: {e}")

    async def spider(self):
        """
        Worker task that fetches, processes, and recurses over URLs.
        """
        while self.running:
            try:
                url, current_depth = await asyncio.wait_for(self.get_url(), timeout=5)
            except asyncio.TimeoutError:
                break # if no URL for 5 seconds, exit loop

            try:
                response = await self.fetch_url(url,current_depth)
                await self.process_response(url, response, current_depth)
            except Exception as e:
                logger.error(f"Spider encountered error on {url}: {e}")
            finally:
                self.task_done()

    async def run(self, seeds=None):
        """
        Entry point to start the crawl.

        Args:
            seeds (list): Initial list of seed URLs.
        """
        if not seeds:
            seeds = ["https://example.com"]
        
        start_time = time.time()
        logger.info(f"Starting crawl at {time.strftime('%H:%M:%S')}")

        await self.storage.async_init()
        await self.seed_urls(seeds, initial_depth=0)  # add seeds to frontier

        async with aiohttp.ClientSession() as session:
            self.fetcher = Fetcher(session)    # initialize fetcher with the session

            # get the default user agent from the fetcher
            user_agent = self.fetcher.default_agent
            # check robots.txt for all seed hosts
            unique_hosts = {urlparse(url).netloc for url in seeds}
            for host in unique_hosts:
                site_url = f"https://{host}"
                await self.fetcher.check_robots(site_url, useragent=user_agent)

            self.running = True  # execution flag
            spiders = [asyncio.create_task(self.spider()) for _ in range(self.num_spiders)]

            try:
                await asyncio.wait_for(self.frontier.join(), timeout=3600)  # wait for frontier to be empty with a reasonable timeout
            except asyncio.TimeoutError:
                logger.warning("Frontier join timed out, stopping spiders")
            finally:
                self.running = False  # signal spiders to exit

            duration = time.time() - start_time
            logger.info(f"Crawling completed after {duration:.2f} seconds")

            await asyncio.gather(*spiders, return_exceptions=True)
            await self.storage.commit()

            logger.info("Crawling finished.")
            logger.info(f"Total seen URLs: {len(self.seen)}")
            logger.info(f"Total successfully visited URLs: {len(self.visited)}")