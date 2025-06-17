"""
will implement async crawler to avoid blocking on requests.

max_concurrency will set the maximum number of concurrent 
tasks that can run at the same time.

will use asyncio semaphore for concurrency control, will ensure 
only max_concurrency async tasks are allowed to run in parallel.

"""
import asyncio
import time
from urllib.parse import urlparse
from collections import defaultdict

from fetcher import Fetcher  # Eva
from parser import Parser    # Eva
from storage import Storage  # Francesca


class Scheduler:
    def __init__(self, max_concurrency=5):
        self.frontier = asyncio.Queue()
        self.visited = set()    # tracks which URLs have been processed
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency) # limits how many fetches can happen at once

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

    def get_hostname(self, url):
        """
        Need to control crawl behavior
        """
        return urlparse(url).netloc
            

    async def seed_url(self, url):
        """
        Seeds the initial URL into the queue.
        """

        await self.frontier.put(url)   # this taks waits for queue.put(url) to complete befor moving on

    async def worker(self):
        """
        Worker that pulls a URL, fetches it, parses it, and queues new links.
        """

        while True:
            url = await self.frontier.get()    # waits asynchronously for a URL to be available in the queue

            if url in self.visited:
                self.frontier.task_done()
                continue

            self.visited.add(url)   # marks the URL as visited

            hostname = self.get_hostname(url)

            async with self.semaphore:
                async with self.host_locks[hostname]:  # request to fetch per host at a time

                    print(f"[Crawler] Fetching {url}")
                    response = await self.fetcher.fetch(url) 

            self.frontier.task_done()

            if not response:
                continue  # skip if fetch failed or blocked by robots.txt
                
            content, final_url, status = response

            if status != 200 or not content:
                continue

            if response:
                try:
                    self.storage.save_page(final_url, content)
                    print(f"[saved] {final_url}")

                except Exception as e:
                    print(f"[error saving] {final_url}: {e}")

                # extract new links and queue them
                links = self.parser.extract_links(content, final_url)  

                for link in links:
                    if link not in self.visited:
                        await self.frontier.put(link)  # if not visited add to queue

            
        

    async def run(self):
        """
        Starts the crawling loop with multiple concurrent workers.
        Creates num_workers number of tasks.
        Each one runs self.worker(fetcher, parser)
        Uses asyncio.create_task() to run them concurrently
        """

        tasks = [
            asyncio.create_task(self.worker()) for _ in range(self.max_concurrency)
            ]

        await self.frontier.join()  # wait for all items in queue to be fully processed

        for t in tasks:
            t.cancel()  # cancel all workers after done so it doesn't run forever

        print(f"[Crawler] Finished. Total pages visited: {len(self.visited)}")

