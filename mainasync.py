import asyncio
import logging
from scheduler import Scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

async def main():
    logging.info("Starting scheduler...")
    max_concurrency = 5
    num_spiders = 3
    scheduler = Scheduler(max_concurrency, num_spiders)

    # Inizializza lo storage (se non lo fai nel costruttore dello Scheduler)
    await scheduler.storage.async_init()

    logging.info("Running scheduler with seeds.")
    await scheduler.run(seeds=["https://www.python.org"])
    logging.info("Scheduler finished.")

if __name__ == "__main__":
    asyncio.run(main())
