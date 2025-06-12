# Web Crawler

## Overview
In this project aweb crawler is implemented for the UniTs **Information Retrieval** exam. The crawler will be built in Python, and will be able to:
* Visit a web page
* Download the page
* Parse the HTML and extract links

---

## Best practices

**Fairness**:

* Respect robots.txt (a file that tells bots what they’re allowed to crawl).

* Don’t overload websites with too many requests too fast (called politeness).

* Add a delay between requests to the same site.

**Robustness**:

* Handle broken links, timeouts, or pages that don’t respond.

* Retry or skip gracefully without crashing.

---

## Project Structure

```
.
├── main.py          # Entry point — orchestrates all components
├── fetcher.py       # Downloads pages, handles robots.txt, headers, errors
├── parser.py        # Parses HTML, extracts and normalizes links
├── scheduler.py     # Manages URL queue, visited set, and flow
├── storage.py       # Saves pages and metadata, handles freshness and deduplication
```
---

## Authors

Francesca Bazzo

Eva Fumo

Alice Macuz
