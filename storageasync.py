#   ----- Libraries and resources to import -----

#   Libraries to import 
import aiofiles
import asyncio
import json
import time
import os
import numpy as np 
import nltk 
from collections import Counter
from simhash import Simhash 
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV
import logging

logger = logging.getLogger(__name__)

#   Functions imported from other modules of the project
#from fetcher import fetch 
#from parser import parse_page_tags_all 

from fetcherasync import fetch 
from parser import parse_page_tags_all


#   Check if the necessary NLTK resources have been downloaded. Otherwise, download them.
for resource in ["stopwords", "wordnet", "punkt", "averaged_perceptron_tagger", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)



#   ----- Constants -----


#   Lambda mapping for different page types 
LAMBDA_BY_TYPE = {
    "frequent": 2.0,
    "occasional": 1.0,
    "static": 0.05,
    "default": 0.5
}

#   Words to remove during preprocessing 
NUMBER_WORDS = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']



#   ----- Utility functions  -----

def get_wordnet_pos(tag):
    """
    Maps POS tag to WordNet format.
    Input: Penn Treebank POS tag (string)
    Output: WordNet-compatible tag (constant)
    """
    if tag.startswith('J'): return ADJ
    if tag.startswith('V'): return VERB
    if tag.startswith('N'): return NOUN
    if tag.startswith('R'): return ADV
    return NOUN

def preprocess_sync(text):
    """
    Synchronous text preprocessing.
    Input: Raw text (string)
    Output: List of lemmatized, filtered tokens (list of strings)
    """
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalpha() and w not in stop_words and w not in NUMBER_WORDS]
    tagged_words = pos_tag(words)
    return [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged_words]

async def preprocess(text):
    
    return await asyncio.to_thread(preprocess_sync, text)

async def content_page(url):
    from fetcherasync import fetch
    from parser import parse_page_tags_all
    html = await fetch(url)
    text_from_page = parse_page_tags_all(html)
    return ' '.join(text_from_page)

def compute_fingerprint(text):
    # Questo rimane sync, chiamare preprocess async esternamente se serve
    words = preprocess_sync(text)
    return Simhash(words).value

def hamming_distance(fp1, fp2):
    x = (fp1 ^ fp2) & ((1 << 64) - 1)
    distance = 0
    while x:
        distance += 1
        x &= x - 1
    return distance

def compute_age(lambda_, t):
    if lambda_ == 0:
        return t
    return (t + lambda_ * np.exp(-lambda_ * t) - 1) / lambda_

def calculate_page_type(content, url=""):
    # stessa funzione di prima
    content = content.lower()
    url = url.lower()
    if "guardian" in url or "cnn.com" in url or "bbc.com" in url:
        if "live" in content and "update" in content:
            return "frequent"
    if "wikipedia.org/wiki/" in url:
        return "static"
    if "live" in url or "breaking" in url:
        return "frequent"
    if any(k in url for k in ["calendar", "event", "workshop", "conference"]):
        return "occasional"
    if any(k in url for k in ["about", "privacy", "contact", "terms"]):
        return "static"
    frequent_keywords = ["breaking news", "live updates", "as it happens", "developing story"]
    occasional_keywords = ["calendar", "workshop", "conference", "event", "seminar"]
    static_keywords = ["contact us", "about us", "privacy policy", "company info", "terms of service"]
    freq_count = sum(kw in content for kw in frequent_keywords)
    occas_count = sum(kw in content for kw in occasional_keywords)
    static_count = sum(kw in content for kw in static_keywords)
    if freq_count >= 2:
        return "frequent"
    if occas_count >= 1:
        return "occasional"
    if static_count >= 1:
        return "static"
    return "default"

def from_gap_encoding(gaps):
    if not gaps:
        return []
    positions = [gaps[0]]
    for gap in gaps[1:]:
        positions.append(positions[-1] + gap)
    return positions

def to_gap_encoding(positions):
    if not positions:
        return []
    return [positions[0]] + [positions[i] - positions[i-1] for i in range(1, len(positions))]

class Storage:

    def __init__(self, pages_file="data/pages.json", index_file="data/inverted_index.json"):
        self.pages_file = pages_file
        self.index_file = index_file
        self.pages = self._load_json_sync(self.pages_file) or {}
        self.inverted_index = self._load_json_sync(self.index_file) or {}
        self._lock = asyncio.Lock()  # Per salvataggi async

    async def _load_json_async(self, filename):
        if not os.path.exists(filename):
            return None
        try:
            async with aiofiles.open(filename, "r", encoding="utf-8") as f:
                content = await f.read()
            # Parse json in thread pool per sicurezza
            return await asyncio.to_thread(json.loads, content)
        except Exception as e:
            logger.error(f"Failed to load JSON file {filename}: {e}")
            return None

    async def _save_json_async(self, filename, data):
        async with self._lock:
            try:
                async with aiofiles.open(filename, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(data, indent=2))
            except Exception as e:
                logger.error(f"Failed to save JSON file {filename}: {e}")

    async def save_page(self, url, content):
        async with self._lock: 
            now = time.time()
            page_type = calculate_page_type(content, url)
            fingerprint = compute_fingerprint(content)
            logger.info(f"Saving page: {url}")

            self.pages[url] = {
                "fingerprint": fingerprint,
                "page_type": page_type,
                "last_fetch": now
            }

            await self.index_terms(url, content, lock_acquired=True)

    async def index_terms(self, url, content, lock_acquired=False):
        if not lock_acquired:
            async with self._lock:
                await self._index_terms_internal(url, content)
        else:
            await self._index_terms_internal(url, content)

    async def _index_terms_internal(self, url, content):
        if url not in self.pages:
            raise ValueError(f"URL {url} not found in pages")

        words = await preprocess(content)
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

        await self._save_json_async(self.index_file, self.inverted_index)

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
        async with self._lock:
            await asyncio.gather(
                self._save_json_async(self.pages_file, self.pages),
                self._save_json_async(self.index_file, self.inverted_index)
            )
        logger.info("Files pages.json and inverted_index.json saved!")











