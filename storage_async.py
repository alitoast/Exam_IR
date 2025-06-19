"""
Storage module (Asynchronous)

The Storage module provides persistent management and indexing of web page data. 
It handles loading and saving page metadata and an inverted index from JSON files asynchronously with 
proper locking to ensure data consistency during concurrent access. The module supports operations 
such as saving new page content, updating the inverted index with term frequencies and positions, 
checking for page refresh needs based on custom aging heuristics, and detecting near-duplicate pages 
via Simhash fingerprinting. 
It acts as the core component for storing, retrieving, and maintaining up-to-date page information 
efficiently.

"""

#   ----- Libraries and resources to import -----

#   Libraries to import 
import aiofiles
import asyncio
import json
import time
import os
from collections import Counter
import logging

from utils_async import calculate_page_type, compute_fingerprint, preprocess, to_gap_encoding, compute_age, hamming_distance

# logging set up
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", # timestamp, level, message
    datefmt="%H:%M:%S",
    filename="crawler_storage.log",  # log file
    filemode="a"  # append to the log file ('w' to overwrite)
)

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

    Methods:
        __init__(pages_file, index_file):
            Synchronous constructor that sets file paths and initializes empty dictionaries for 
            `pages` and `inverted_index`. 
            It creates an asynchronous lock to manage concurrent write operations.

        async_init():
            Asynchronously loads data from JSON files.

        _load_json_async(filename):
            Loads JSON file asynchronously, returns dict or None.

        _save_json_async(filename, data):
            Saves data as JSON asynchronously under lock.

        save_page(url, content):
            Saves a page's metadata and indexes its terms asynchronously.

        index_terms(url, content, lock_acquired=False):
            Indexes terms of a page asynchronously, optionally assuming lock is held.

        _index_terms_internal(url, content):
            Internal method to perform term indexing and update inverted index.

        get_page(url):
            Retrieves metadata dictionary for a given URL.

        get_page_type(url):
            Retrieves the type classification of a stored page.

        get_last_fetch(url):
            Retrieves the last fetch timestamp of a stored page.

        get_tf(term):
            Retrieves term frequency postings for a term.

        needs_refresh(url):
            Determines if a page should be re-fetched based on its age and type.

        is_near_duplicate(content, threshold=5):
            Checks if the content is near-duplicate of any stored page.

        commit():
            Saves pages metadata and inverted index to disk asynchronously.
    """

    def __init__(self, pages_file="data/pages.json", index_file="data/inverted_index.json"):
        self.pages_file = pages_file
        self.index_file = index_file
        self.pages = {}
        self.inverted_index = {}
        self.dirty = False 
        self._lock = asyncio.Lock()

    async def async_init(self): 
        self.pages = await self._load_json_async(self.pages_file) or {}
        self.inverted_index = await self._load_json_async(self.index_file) or {}
        logger.info(f"Storage initialized. Loaded {len(self.pages)} pages and {len(self.inverted_index)} terms.")

    async def _load_json_async(self, filename):
        if not os.path.exists(filename):
            return None
        logger.debug(f"[LOAD] Attempting to load JSON file: {filename}")
        try:
            async with aiofiles.open(filename, "r", encoding="utf-8") as f:
                content = await f.read()
            return await asyncio.to_thread(json.loads, content)
        except Exception as e:
            logger.error(f"Failed to load JSON file {filename}: {e}")
            return None

    async def _save_json_async(self, filename, data):
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
        if not os.path.exists(filename):
            return None
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Errore nel caricamento di {filename}: {e}")
            return None

    async def save_page(self, url, content):
        start = time.perf_counter() #   debug
        logger.info(f"[PAGE] Saving page: {url}")
        async with self._lock: 
            mid = time.perf_counter()
            logger.info(f"[PAGE] Lock acquisito dopo {mid - start:.3f} secondi")
            now = time.time()
            page_type = calculate_page_type(content, url)
            fingerprint = compute_fingerprint(content)
            logger.info(f"[PAGE] Computed fingerprint e page_type dopo {time.perf_counter() - mid:.3f} secondi")    
            self.pages[url] = {
                "fingerprint": fingerprint,
                "page_type": page_type,
                "last_fetch": now
            }
            self.dirty = True 
            await self.index_terms(url, content, lock_acquired=True)
            logger.info(f"[PAGE] Indexed terms e aggiornato struttura dopo {time.perf_counter() - mid:.3f} secondi")


    async def index_terms(self, url, content, lock_acquired=False):
        logger.debug(f"[INDEX] Starting index for {url}, lock_acquired={lock_acquired}")
        if not lock_acquired:
            async with self._lock:
                await self._index_terms_internal(url, content)
                self.dirty = True 
        else:
            await self._index_terms_internal(url, content)


    async def _index_terms_internal(self, url, content):
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
        # await self._save_json_async(self.index_file, self.inverted_index)

    def get_page(self, url):
        return self.pages.get(url)

    def get_page_type(self, url):
        page = self.pages.get(url)
        if page:
            return page.get("page_type")
        return None

    def get_last_fetch(self, url):
        page = self.pages.get(url)
        if page:
            return page.get("last_fetch")
        return None

    def get_tf(self, term):
        return self.inverted_index.get(term, {})

    def needs_refresh(self, url):
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
        if not self.dirty:
            logger.info("[COMMIT] Nessuna modifica da salvare. Commit saltato.")
            return        
        logger.info(f"[DEBUG] Tentativo di commit. pages: {len(self.pages)}, index: {len(self.inverted_index)}")
        async with self._lock:
            logger.debug("Inizio salvataggio pages_file")
            await self._save_json_async(self.pages_file, self.pages)
            logger.debug("pages_file salvato, inizio salvataggio index_file")
            await self._save_json_async(self.index_file, self.inverted_index)
            logger.debug("index_file salvato")

        logger.info("Files pages.json and inverted_index.json saved!")












