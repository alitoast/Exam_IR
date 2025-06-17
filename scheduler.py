"""
will implement async crawler to avoid blocking on requests.

max_concurrency will set the maximum number of concurrent 
tasks that can run at the same time.

will use asyncio semaphore for concurrency control, will ensure 
only max_concurrency async tasks are allowed to run in parallel.

max_concurrent: Limits simultaneous fetches (via semaphore)

num_spiders: Number of async crawling tasks running concurrently

"""
import asyncio
import time
from urllib.parse import urlparse
from collections import defaultdict

from fetcher import Fetcher  # Eva
from parser import Parser    # Eva
from storage import Storage  # Francesca


class Scheduler:
    def __init__(self, max_concurrency=5, num_spiders=5):
        self.frontier = asyncio.Queue()                         # URLs to crawl
        self.visited = set()                                    # tracks visited URLs 
        self.semaphore = asyncio.Semaphore(max_concurrency)     # limits max parallel fetches

        self.max_concurrency = max_concurrency
        self.num_spiders = num_spiders

        self.fetcher = Fetcher()
        self.parser = Parser()
        self.storage = Storage()

        """
        Need to choose what's best either this with fixes cause is synchronous. 
        Probably best to keeep the method later but
        needs check cause r equires an async context to call this method. 
        and needs to be called explicitly from outside after instantiation.

        
        # populate the frontier
        for seed in seeds:
            self.frontier.put_nowait(seed)
        """

    async def seed_url(self, url):
        """
        Seeds the initial URL into the queue.
        """

        await self.frontier.put(url)   # this taks waits for queue.put(url) to complete befor moving on

    def get_hostname(self, url):
        """
        Extract the hostname from a URL.
        """
        return urlparse(url).netloc
    
    async def add_url(self, url):
        """
        Add a new URL to the frontier if it hasn't been visited.
        """
        if url not in self.visited:
            await self.frontier.put(url)
            self.visited.add(url)
    
    async def get_url(self):
        """
        Get the next URL from the frontier.
        Wait asynchronously for a URL to be available in the queue.
        """
        return await self.frontier.get()
    
    def task_done(self):
        """
        Mark the current task as completed in the frontier.
        """
        self.frontier.task_done()
         
    
    async def spider(self):
        """
        Spider that pulls a URL, fetches it, parses it, and queues new links.
        """

        while True:

            url = await self.get_url()

            if url in self.visited:
                self.task_done()
                continue

            self.visited.add(url)   # marks the URL as visited
            hostname = self.get_hostname(url)

            # enforce both global fetch concurrency and per-host politeness
            async with self.semaphore:
                async with self.host_locks[hostname]:
                    print(f"[Crawler] Fetching {url}")
                    response = await self.fetcher.fetch(url) 

            self.task_done()

            if not response:
                continue  # skip if fetch failed or blocked by robots.txt
                
            content, final_url, status = response

            # skip non-successful responses or empty content
            if status != 200 or not content:
                continue

            await self.storage.save_page(final_url, content)
            print(f"[saved] {final_url}")

            
            # extract and enqueue links found in the page
            links = self.parser.extract_links(content, final_url)  

            for link in links:
                await self.add_url(link)       

    async def run(self):
        """
        Starts the crawling loop with multiple concurrent spiders.
        Creates num_spiders number of tasks.
        Each one runs self.worker(fetcher, parser)
        Uses asyncio.create_task() to run them concurrently
        """

        spiders = [
            asyncio.create_task(self.spider()) for _ in range(self.num_spiders)
            ]

        await self.frontier.join()  # wait for all items in queue to be fully processed

        for s in spiders:
            s.cancel()  # cancel all spiders after done so it doesn't run forever

        print(f"[Crawler] Finished. Total pages visited: {len(self.visited)}")

