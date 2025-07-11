# main.py

import asyncio
from scheduler import Scheduler
from config import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

async def main():
    logging.info("Starting scheduler...")
    max_concurrency = 5
    num_spiders = 3
    max_depth = 2
    max_pages_per_domain = 10

    scheduler = Scheduler(max_concurrency, num_spiders, max_depth, max_pages_per_domain)

    logging.info("Running scheduler with seeds.")
    await scheduler.run(seeds=["https://example.com"])
    logging.info("Scheduler finished.")

if __name__ == "__main__":
    asyncio.run(main())
