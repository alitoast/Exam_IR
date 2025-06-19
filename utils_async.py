"""
Utils Module (Asynchronous)
===========================

This module provides asynchronous utility functions for NLP preprocessing, content fingerprinting,
gap encoding, and heuristic classification. These functions support the content fetching, parsing,
and indexing pipeline of a crawler or search engine.

Dependencies:
-------------
- `nltk`                : Tokenization, stopwords, POS tagging, lemmatization
- `numpy`               : Numerical aging model
- `simhash`             : Duplicate detection via fingerprinting
- `asyncio`             : Async execution handling
- `logging`             : Debug and performance logging
"""


#   Libraries to import 
import asyncio
import numpy as np 
import nltk 
import time 
from collections import Counter
from simhash import Simhash 
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV
import logging

logger = logging.getLogger(__name__)


# Functions imported from other modules of the project
from fetcher import Fetcher 
from parser import Parser 

fetcher = Fetcher(None) 
parser = Parser() 

# Ensure NLTK resources are available
for resource in ["stopwords", "wordnet", "punkt", "averaged_perceptron_tagger", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)


#   Words to remove during preprocessing 
NUMBER_WORDS = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']


def get_wordnet_pos(tag):
    """
    Maps POS tag to WordNet format.
    Input: 
        tag (str): POS tag in Penn Treebank format.
    Output:
        WordNet POS tag constant (e.g., NOUN, VERB, ADJ, ADV).
    Description:
        Converts common Penn Treebank POS tags to the corresponding
        WordNet POS constants used for lemmatization.
    """
    if tag.startswith('J'): return ADJ
    if tag.startswith('V'): return VERB
    if tag.startswith('N'): return NOUN
    if tag.startswith('R'): return ADV
    return NOUN


def preprocess_sync(text):
    """
    Synchronously preprocesses text: tokenization, stopword removal, POS tagging, lemmatization.

    Args:
        text (str): Raw input text.

    Returns:
        list[str]: Cleaned and lemmatized token list.
    """
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalpha() and w not in stop_words and w not in NUMBER_WORDS]
    tagged_words = pos_tag(words)
    return [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged_words]


async def preprocess(text):
    """
    Asynchronously preprocesses text by running the synchronous pipeline in a thread.

    Args:
        text (str): Raw input text.

    Returns:
        list[str]: Cleaned and lemmatized tokens..
    """
    start = time.perf_counter()
    result = await asyncio.to_thread(preprocess_sync, text) 
    end = time.perf_counter() 
    logger.info(f"[PREPROCESS] Text processed in {end - start:.3f} seconds")
    return result


async def content_page(url):
    """
    Fetches the HTML page and extracts visible textual content.

    Args:
        url (str): Target URL to crawl.

    Returns:
        str: Concatenated page content text.
    """
    start = time.perf_counter()
    html = await fetcher.fetch(url)
    mid = time.perf_counter() 
    text_from_page = parser.parse_page_tags_all(html)
    end = time.perf_counter()
    logger.info(f"[FETCH] Fetch: {mid - start:.3f}s, Parse: {end - mid:.3f}s")
    return ' '.join(text_from_page)


def compute_fingerprint(text):
    """ 
    Computes a Simhash fingerprint from the provided text.

    Args:
        text (str): Input text.

    Returns:
        int: 64-bit Simhash value.
    """
    words = preprocess_sync(text)
    return Simhash(words).value


def hamming_distance(fp1, fp2):
    """
    Calculates the Hamming distance between two fingerprints.

    Args:
        fp1 (int): First Simhash value.
        fp2 (int): Second Simhash value.

    Returns:
        int: Bitwise difference count.
    """
    x = (fp1 ^ fp2) & ((1 << 64) - 1)
    distance = 0
    while x:
        distance += 1
        x &= x - 1
    return distance


def compute_age(lambda_, t):
    """
    Applies exponential decay to estimate content freshness.

    Args:
        lambda_ (float): Decay rate.
        t (float): Time since last fetch (e.g., days).

    Returns:
        float: Age score reflecting staleness.
    """
    if lambda_ == 0:
        return t
    return (t + lambda_ * np.exp(-lambda_ * t) - 1) / lambda_


def calculate_page_type(content, url=""):
    """
    Heuristically determines page type based on content and URL.

    Args:
        content (str): Visible text from page.
        url (str, optional): Page URL.

    Returns:
        str: One of "frequent", "occasional", "static", or "default".
    """
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
    """
    Converts gap-encoded token positions into absolute positions.

    Args:
        gaps (list[int]): List of gap-encoded positions.

    Returns:
        list[int]: Reconstructed absolute positions.
    """
    if not gaps:
        return []
    positions = [gaps[0]]
    for gap in gaps[1:]:
        positions.append(positions[-1] + gap)
    return positions


def to_gap_encoding(positions):
    """
    Converts absolute token positions to gap-encoded format.

    Args:
        positions (list[int]): Token positions in absolute format.

    Returns:
        list[int]: Gap-encoded representation.
    """
    if not positions:
        return []
    return [positions[0]] + [positions[i] - positions[i-1] for i in range(1, len(positions))]