import asyncio
import nltk
from storage import Storage
from parser import Parser
from fetcher import Fetcher
from scheduler import Scheduler
from config import setup_logging
import utils_async
import logging


setup_logging()
logger = logging.getLogger(__name__)


async def main():
    # Download necessary NLTK data
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

    # Initialize components
    storage = Storage()
    parser = Parser()
    fetcher = Fetcher()
    scheduler = Scheduler(max_concurrency=10, num_spiders=5)

    # Optionally: add initial seed URLs
    seed_urls = [
        "https://example.com",
        "https://news.ycombinator.com"
    ]
    await scheduler.seed_urls(seed_urls)

    # Start the crawling process
    try:
        await scheduler.run(seed_urls)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        # Commit storage before exit
        await storage.commit()

if __name__ == "__main__":
    asyncio.run(main())