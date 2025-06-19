import asyncio
from scheduler import Scheduler
from mock import Storage  # Import the mock storage

async def test_scheduler():
    # Configure the scheduler with the mock storage
    scheduler = Scheduler(max_concurrency=5, num_spiders=2, storage=Storage())
    seeds = ["https://example.com", "https://example.com/fail"]

    await scheduler.run(seeds)

    print("Seen URLs:", scheduler.seen)
    print("Visited URLs:", scheduler.visited)

    # Assertions
    assert "https://example.com" in scheduler.visited, "Expected main URL to be visited"
    assert "https://example.com/child" in scheduler.visited, "Expected child URL to be visited"

    # This should not be in visited if it failed
    assert "https://example.com/fail" not in scheduler.visited, "Failing URL should not be in visited"
    assert "https://example.com/fail" in scheduler.seen, "Failing URL should be in seen"

    # Check counts
    assert len(scheduler.visited) == 2, "Expected 2 URLs to be successfully visited"
    assert len(scheduler.seen) >= 3, "Expected at least 3 URLs to be seen (including retries and children)"

    print("All tests passed.")

# Run the test
if __name__ == "__main__":
    asyncio.run(test_scheduler())