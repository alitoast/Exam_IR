"""
Storage Module (Asynchronous)
=============================

Description:
------------
This module provides asynchronous tools for persistent storage and indexing of web page data, 
designed to support a web crawler.
It maintains structured metadata about pages — such as timestamps, classification types, link structures, and Simhash fingerprints —
and constructs an inverted index for fast term lookup.
All I/O operations are non-blocking and synchronized with locking mechanisms to ensure consistency during concurrent access.

Key Features:
-------------
- Asynchronous loading and saving of metadata and inverted index (JSON-based)
- Fine-grained concurrency control using `asyncio.Lock`
- Page classification and aging for adaptive refresh strategies
- Simhash-based near-duplicate detection using Hamming distance
- Gap-encoded positional inverted indexing
- Integration with external preprocessing and parsing modules
- Persistent commit mechanism to serialize in-memory state to disk

Dependencies:
-------------
- `aiofiles`            : Asynchronous file reading and writing
- `asyncio`             : Core library for async event loop and locking
- `json`                : Serialization and deserialization of data
- `time`                : Timestamping for page fetch and age computation
- `os`                  : File system operations and path management
- `collections.Counter` : Term frequency counting
- `logging`             : Structured logging and debug output
- `parser`              : Custom HTML parser for extracting links
- `utils_async`         : Utilities for preprocessing, Simhash, encoding, and age modeling

Use Cases:
----------
- Asynchronous, fault-tolerant storage backend for a web crawler
- Detection and prevention of near-duplicate content indexing
- Efficient inverted indexing for term-based retrieval systems
- Smart refresh scheduling of pages based on content volatility
- Persistent and recoverable storage of crawl metadata across sessions
"""

#   ----- Libraries and dependencies -----

import aiofiles
import asyncio
import json
import time
import os
from collections import Counter
import logging
from parser import Parser 

from utils_async import calculate_page_type, compute_fingerprint, preprocess, to_gap_encoding, compute_age, hamming_distance

parser = Parser() 

logger = logging.getLogger(__name__)


#   Lambda mapping for different page types 
LAMBDA_BY_TYPE = {
    "frequent": 2.0,
    "occasional": 1.0,
    "static": 0.05,
    "default": 0.5
}



class Storage:

    """
    Asynchronous storage manager for pages metadata and inverted index.

    Attributes:
        pages_file (str): Path to the JSON file storing pages metadata.
        index_file (str): Path to the JSON file storing inverted index.
        pages (dict): In-memory dictionary of pages metadata.
        inverted_index (dict): In-memory inverted index mapping terms to postings.
        _lock (asyncio.Lock): Async lock to protect concurrent file access.
    """

    def __init__(self, pages_file="data/pages.json", index_file="data/inverted_index.json"):
        """
        Initializes the Storage object with file paths and in-memory data structures.
        Args:
            pages_file (str): Path to the pages metadata JSON file.
            index_file (str): Path to the inverted index JSON file.
        Returns:
            None
        """
        self.pages_file = pages_file
        self.index_file = index_file
        self.pages = {}
        self.inverted_index = {}
        self.dirty = False 
        self._lock = asyncio.Lock()
        os.makedirs(os.path.dirname(self.pages_file), exist_ok=True)

    async def async_init(self): 
        """
        Asynchronously loads the pages metadata and inverted index from disk.

        Returns:
            None
        """
        self.pages = await self._load_json_async(self.pages_file) or {}
        self.inverted_index = await self._load_json_async(self.index_file) or {}
        logger.info(f"Storage initialized. Loaded {len(self.pages)} pages and {len(self.inverted_index)} terms.")

    async def _load_json_async(self, filename):
        """
        Loads a JSON file asynchronously.

        Args:
            filename (str): Path to the JSON file.

        Returns:
            dict or None: Parsed JSON data, or None if file is missing or invalid.
        """
        if not os.path.exists(filename):
            return None
        logger.debug(f"[LOAD] Attempting to load JSON file: {filename}")
        try:
            async with aiofiles.open(filename, "r", encoding="utf-8") as f:
                content = await f.read()
            # Se il file è vuoto o contiene solo spazi bianchi, inizializza dict vuoto
            if not content.strip():
                logger.warning(f"[LOAD] {filename} is empty — initializing empty dict")
                return {}
            return await asyncio.to_thread(json.loads, content)
        except json.JSONDecodeError as e:
            logger.warning(f"[LOAD] Invalid JSON in {filename} ({e}) — initializing empty dict")
            return {}
        except Exception as e:
            logger.error(f"Failed to load JSON file {filename}: {e}")
            return {}

    async def _save_json_async(self, filename, data):
        """
        Saves a dictionary as a JSON file asynchronously.

        Args:
            filename (str): Path to the JSON file.
            data (dict): Data to serialize and write.

        Returns:
            None
        """
        if not data:
            logger.info(f"[DEBUG] {filename} vuoto. Scrittura saltata.")
            return
        logger.debug(f"[SAVE] Attempting to save {filename}, records: {len(data)}")
        try:
            logger.debug(f"Inizio scrittura file {filename} (dimensione dati: {len(json.dumps(data))} bytes)")
            async with aiofiles.open(filename, "w", encoding="utf-8") as f:
                await f.write(json.dumps(data, indent=2))
            logger.debug(f"Scrittura file {filename} completata")
        except Exception as e:
            logger.error(f"Failed to save JSON file {filename}: {e}")


    def _load_json_sync(self, filename):
        """
        Loads a JSON file synchronously.

        Args:
            filename (str): Path to the JSON file.

        Returns:
            dict or None: Parsed JSON data, or None if file is missing or invalid.
        """
        if not os.path.exists(filename):
            return None
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Errore nel caricamento di {filename}: {e}")
            return None

    async def save_page(self, url, content):
        """
        Saves page metadata and indexes terms for a given page.
        Args:
            url (str): page URL.
            content (str): text content of the page.
        Returns:
            None
        """
        start = time.perf_counter() 
        logger.info(f"[PAGE] Saving page: {url}")
        async with self._lock: 
            mid = time.perf_counter()
            logger.info(f"[PAGE] Lock acquired after {mid - start:.3f} seconds")
            now = time.time()
            page_type = calculate_page_type(content, url)
            fingerprint = compute_fingerprint(content)
            outlinks = parser.extract_links(content, url) 
            logger.info(f"[PAGE] Computed fingerprint and page_type after {time.perf_counter() - mid:.3f} seconds")    
            self.pages[url] = {
                "fingerprint": fingerprint,
                "page_type": page_type,
                "last_fetch": now, 
                "outlinks": outlinks or [] 
            }   
            self.dirty = True 
            await self.index_terms(url, content, lock_acquired=True)
            logger.info(f"[PAGE] Indexed terms updated after {time.perf_counter() - mid:.3f} seconds")

    def get_outlinks(self, url):
        """
        Retrieves the list of outlinks for a given page URL.

        Args:
            url (str): Page URL.

        Returns:
            list: List of outlink URLs, or empty list if not found.
        """    
        page = self.pages.get(url)
        if page:
            return page.get("outlinks", [])
        return []


    async def index_terms(self, url, content, lock_acquired=False):
        """
        Indexes terms of a page content asynchronously.

        Args:
            url (str): Page URL.
            content (str): Raw page content.
            lock_acquired (bool): If True, assumes the lock is already held.

        Returns:
            None
        """
        logger.debug(f"[INDEX] Starting index for {url}, lock_acquired={lock_acquired}")
        if not lock_acquired:
            async with self._lock:
                await self._index_terms_internal(url, content)
                self.dirty = True 
        else:
            await self._index_terms_internal(url, content)


    async def _index_terms_internal(self, url, content):
        """
        Internal method for term extraction and inverted index update.

        Args:
            url (str): Page URL.
            content (str): Raw page content.

        Returns:
            None
        """
        if url not in self.pages:
            raise ValueError(f"URL {url} not found in pages")

        words = await preprocess(content)
        logger.debug(f"[INDEX] Indexing {len(words)} words for {url}")
        tf = Counter()
        positions = {}

        for i, word in enumerate(words):
            tf[word] += 1
            positions.setdefault(word, []).append(i)

        for word in positions:
            positions[word] = to_gap_encoding(positions[word])

        # Remove old entries
        for term in list(self.inverted_index.keys()):
            if url in self.inverted_index[term]:
                del self.inverted_index[term][url]
                if not self.inverted_index[term]:
                    del self.inverted_index[term]

        for term in tf:
            if term not in self.inverted_index:
                self.inverted_index[term] = {}
            self.inverted_index[term][url] = {
                "tf": tf[term],
                "positions": positions[term]
            }

        logger.debug(f"[INDEX] Updated index with {len(tf)} unique terms for {url}")


    def needs_refresh(self, url):
        """
        Determines whether a page needs to be re-fetched based on its age and type.

        Args:
            url (str): Page URL.

        Returns:
            bool: True if the page should be refreshed, else False.
        """
        page = self.pages.get(url)
        if not page:
            return True

        last_fetch = page.get("last_fetch")
        if last_fetch is None:
            return True

        page_type = page.get("page_type", "default")

        now = time.time()
        time_in_days = (now - last_fetch) / 86400
        lambda_ = LAMBDA_BY_TYPE.get(page_type, 0.5)
        threshold = 1 / lambda_

        age = compute_age(lambda_, time_in_days)
        return age > threshold

    def is_near_duplicate(self, content, threshold=5):
        """
        Checks if the provided content is a near-duplicate of any stored page.

        Args:
            content (str): Page content to check.
            threshold (int): Hamming distance threshold for duplicate detection.

        Returns:
            tuple: (bool, str or None) — True and matched URL if duplicate is found, else (False, None).
        """
        new_fp = compute_fingerprint(content)
        for url, page_data in self.pages.items():
            fp = page_data.get("fingerprint")
            if fp is None:
                continue
            try:
                d = hamming_distance(new_fp, fp)
                if d <= threshold:
                    return True, url
            except Exception:
                continue
        return False, None

    async def commit(self):
        """
        Asynchronously saves in-memory pages and inverted index to disk if changes were made.

        Returns:
            None
        """
        if not self.dirty:
            logger.info("[COMMIT] No changes to save. Skip commit.")
            return        
        logger.info(f"[DEBUG] Commiting changes. Pages {len(self.pages)}, index terms: {len(self.inverted_index)}")
        async with self._lock:
            logger.debug("Saving pages_file")
            await self._save_json_async(self.pages_file, self.pages)
            logger.debug("pages_file saved, saving index_file")
            await self._save_json_async(self.index_file, self.inverted_index)
            logger.debug("index_file saved")

        logger.info("Files pages.json and inverted_index.json saved!")












