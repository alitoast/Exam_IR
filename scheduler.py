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
        self.visited = set()
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def seed_url(self, url):
        """Seeds the initial URL into the queue."""
        pass

    async def worker(self, fetcher, parser):
        """Worker that pulls a URL, fetches it, parses it, and queues new links."""
        pass

    async def run(self, fetcher, parser, num_workers=5):
        """Starts the crawling loop with multiple concurrent workers."""
        pass

