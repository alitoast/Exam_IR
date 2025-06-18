"""
Utils module (Asynchronous)
============================= 
The Helpers module contains utility functions focused on text preprocessing, linguistic analysis, 
and data transformations used throughout the project. 
It provides functions for part-of-speech tagging mapped to WordNet categories, tokenization and 
lemmatization of raw text, calculation of Simhash fingerprints for duplicate detection, and conversion 
utilities for gap-encoded positional data. Additionally, it offers heuristics to classify page types 
based on URL and content keywords, and methods to compute aging metrics for content freshness. 
These helper functions encapsulate reusable logic for natural language processing and data handling 
to support higher-level storage and indexing operations.


"""


#   Libraries to import 
import asyncio
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

from fetcherasync import fetch
from parser import parse_page_tags_all

#   Check if the necessary NLTK resources have been downloaded. Otherwise, download them.
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
    Synchronously preprocesses text by tokenizing, filtering, POS tagging,
    and lemmatizing.

    Input:
        text (str): Raw input text.

    Output:
        list of str: Preprocessed and lemmatized tokens.

    Description:
        Performs tokenization, removes stopwords and numeric words, tags POS,
        and lemmatizes tokens to their base form.
    """
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalpha() and w not in stop_words and w not in NUMBER_WORDS]
    tagged_words = pos_tag(words)
    return [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged_words]


async def preprocess(text):
    """
    Asynchronously preprocesses text by running synchronous preprocessing
    in a separate thread.

    Input:
        text (str): Raw input text.

    Output:
        list of str: Preprocessed and lemmatized tokens.

    Description:
        Wraps synchronous preprocessing to avoid blocking the async event loop.
    """
    return await asyncio.to_thread(preprocess_sync, text)


async def content_page(url):
    """
    Fetches HTML content from a URL asynchronously and parses text content.

    Input:
        url (str): URL of the webpage.

    Output:
        str: Concatenated textual content extracted from the page.

    Description:
        Uses asynchronous fetch and synchronous parser to retrieve and process
        page content for indexing or analysis.
    """
    html = await fetch(url)
    text_from_page = parse_page_tags_all(html)
    return ' '.join(text_from_page)


def compute_fingerprint(text):
    """ 
        Computes a 64-bit Simhash fingerprint for a given text.

    Input:
        text (str): Text to fingerprint.

    Output:
        int: Simhash fingerprint value.

    Description:
        Uses synchronous preprocessing to tokenize and lemmatize the text,
        then computes a Simhash to represent the text compactly for near-duplicate detection.
    """
    words = preprocess_sync(text)
    return Simhash(words).value


def hamming_distance(fp1, fp2):
    """
    Computes the Hamming distance between two 64-bit integer fingerprints.

    Input:
        fp1 (int): First fingerprint.
        fp2 (int): Second fingerprint.

    Output:
        int: Number of differing bits.

    Description:
        Counts differing bits between two fingerprints to measure similarity.
    """
    x = (fp1 ^ fp2) & ((1 << 64) - 1)
    distance = 0
    while x:
        distance += 1
        x &= x - 1
    return distance


def compute_age(lambda_, t):
    """
    Computes the age of content using an exponential decay model.

    Input:
        lambda_ (float): Decay rate parameter.
        t (float): Time since last fetch (e.g., in days).

    Output:
        float: Computed age score.

    Description:
        Models content freshness with an aging function based on decay lambda.
    """
    if lambda_ == 0:
        return t
    return (t + lambda_ * np.exp(-lambda_ * t) - 1) / lambda_


def calculate_page_type(content, url=""):
    """
    Classifies a page type based on URL and content heuristics.

    Input:
        content (str): Text content of the page.
        url (str): URL of the page (optional).

    Output:
        str: One of "frequent", "occasional", "static", or "default".

    Description:
        Uses keywords and URL patterns to assign a frequency category to the page.
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
    Decodes a list of gap-encoded positions back into absolute positions.

    Input:
        gaps (list of int): Gap-encoded positions.

    Output:
        list of int: Absolute positions.

    Description:
        Reconstructs absolute token positions from gap-encoded format.
    """
    if not gaps:
        return []
    positions = [gaps[0]]
    for gap in gaps[1:]:
        positions.append(positions[-1] + gap)
    return positions


def to_gap_encoding(positions):
    """
    Encodes a list of absolute positions into gap-encoded format.

    Input:
        positions (list of int): Absolute token positions.

    Output:
        list of int: Gap-encoded positions.

    Description:
        Compresses positions by storing gaps between successive positions.
    """
    if not positions:
        return []
    return [positions[0]] + [positions[i] - positions[i-1] for i in range(1, len(positions))]