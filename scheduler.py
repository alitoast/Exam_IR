"""
will implement async crawler to avoid blocking on requests.

max_concurrency will set the maximum number of concurrent 
tasks that can run at the same time.

will use asyncio semaphore for concurrency control, will ensure 
only max_concurrency async tasks are allowed to run in parallel.
"""

import asyncio

class Scheduler:
    def __init__(self, max_concurrency=5):
        self.queue = asyncio.Queue()
        self.visited = set()    # tracks which URLs have been processed
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency) # limits how many fetches can happen at once

    async def seed_url(self, url):
        """Seeds the initial URL into the queue."""
        await self.queue.put(url)   # this taks waits for queue.put(url) to complete befor moving on

    async def worker(self, fetcher, parser):
        """Worker that pulls a URL, fetches it, parses it, and queues new links."""

        while True:
            url = await self.queue.get()    # waits asynchronously for a URL to be available in the queue

            if url in self.visited:
                self.queue.task_done()
                continue

            self.visited.add(url)   # marks the URL as visited
            async with self.semaphore:
                html = await fetcher.fetch(url) # request to fetch the page async

            if html: 
                links = parser.extract_links(url,html)  # page fetched use parser to extract links from url

                for link in links:
                    if link not in self.visited:
                        await self.queue.put(link)  # if not visited add to queue

            self.queue.task_done
        

    async def run(self, fetcher, parser, num_workers=5):
        """Starts the crawling loop with multiple concurrent workers."""
        pass

