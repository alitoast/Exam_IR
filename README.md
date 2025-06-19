# Web Crawler -Informatin Retrieval Project

## Overview

This project implements a **web crawler** in Python as part of the UniTs *Information Retrieval* exam. The crawler is designed to:

* Visit web pages
* Download HTML content
* Parse and extract hyperlinks
* Respect crawling ethics and site-specific rules

---

## Features & Best Practices

### Fairness

* **robots.txt support**: The crawler checks `robots.txt` and follows the site’s crawling rules.
* **Politeness policy**: Adds delays between requests to avoid overwhelming servers.
* **Domain-aware scheduling**: Limits the number of pages fetched per domain.

### Robustness

* Handles broken links, timeouts, and HTTP errors.
* Implements retries on failure, but avoids infinite loops.
* Skips duplicate or near-duplicate content using hashing techniques.

---

## Project Structure

```
.
├── main.py          # Entry point — launches the scheduler and starts crawling
├── fetcher.py       # Fetches pages, handles HTTP headers, robots.txt, and retries
├── parser.py        # Parses HTML and extracts/normalizes links
├── scheduler.py     # Manages the URL frontier, visited pages, and concurrency
├── storage.py       # Saves pages and metadata, checks freshness and deduplication
├── utils_async.py   # Utility functions for async operations
├── config.py        # # Sets up logging to both console and file (crawler.log)
```

---


1. **Requirements**

The following Python packages are required:

* `rfc3986`
* `aiohttp`
* `beautifulsoup4`

You can install them using:

```bash
pip install rfc3986 aiohttp beautifulsoup4
```

---

2. **Run the crawler**:

   ```bash
   python main.py
   ```

2. **Configure seeds and parameters** inside `main.py`.
    
Parameters: 
* max_concurrency (int): Global max concurrent fetches.
* num_spiders (int): Number of concurrent spider tasks.
* max_depth (int): Maximum crawl depth per seed.
* max_pages_per_domain (int): Max pages to crawl per domain.


**Default is:**

 ```bash
    max_concurrency = 5
    num_spiders = 3
    max_depth = 3
    max_pages_per_domain = 10
```

---

## Authors

* Francesca Bazzo
* Eva Fumo
* Alice Macuz
