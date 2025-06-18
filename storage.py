""""
Storage Module 
====================

This module provides utilities and storage management for web page content processing and indexing.
It includes functions for preprocessing text data, computing content fingerprints using Simhash, 
classifying pages by update frequency, and managing an inverted index for efficient term-based retrieval. 
The Storage class manages persistence of pages and the inverted index, and provides freshness and 
near-duplicate detection based on content fingerprints.

Main functionalities:
- Text preprocessing with tokenization, stopword removal, POS tagging, and lemmatization
- Page content fetching and parsing integration with fetcher and parser modules 
- Fingerprint calculation using Simhash for near-duplicate detection
- Page type classification to adapt crawling frequency
- Inverted index creation with gap-encoded word positions for efficient storage
- Storage management with JSON-based persistence


Dependencies 
------------------
To use this module, install: 
- nltk 
- numpy 
- simhash
"""


#   ----- Libraries and resources to import -----

#   Libraries to import 
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


#   Functions imported from other modules of the project:
#   - from fetcher import fetch 
#   - from parser import parse_page_tags_all 

from fetcher import fetch 
from parser import parse_page_tags_all


#   Check if the necessary NLTK resources have been downloaded. Otherwise, download them.
for resource in ["stopwords", "wordnet", "punkt", "averaged_perceptron_tagger", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource)



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
    Input:
        tag (str): POS tag from nltk.pos_tag (Treebank format)

    Output:
        wordnet_tag (str): Corresponding WordNet POS tag (NOUN, VERB, ADJ, ADV)

    Description:
        This function converts POS tags from the Penn Treebank tagset used by NLTK's pos_tag
        to the WordNet POS tags needed by the lemmatizer for accurate lemmatization.
    """
    if tag.startswith('J'): return ADJ
    if tag.startswith('V'): return VERB
    if tag.startswith('N'): return NOUN
    if tag.startswith('R'): return ADV
    return NOUN


def preprocess(text):
    """
    Input:
        text (str): Raw text content of a web page

    Output:
        List[str]: List of lemmatized tokens after stopword, number, and punctuation removal

    Description:
        The function lowercases and tokenizes the input text, removes stop words, 
        number words (e.g. 'one', 'two'), and punctuation. Then, it tags each word with its
        POS and lemmatizes the words using WordNet's lemmatizer to normalize word forms.
    """
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    #   These words are not in the stop words set, but I do not want them in the index, so they 
    #   will be removed 
    words = word_tokenize(text.lower())
    #   Remove stop words, number_words and punctuation 
    words = [w for w in words if w.isalpha() and w not in stop_words and w not in NUMBER_WORDS]
    #   Each words is coupled with their tag 
    tagged_words = pos_tag(words)    
    #   Lemmatization is done using the right grammatical type and then the words are returned 
    return [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged_words]


def content_page(url): 
    """ 
    Input:
        url (str): URL of the web page to fetch

    Output:
        str: Extracted text content of the page

    Description:
        Uses the function fetch() from module fetcher to download HTML content of the given URL,
        then uses the function parse_page_tags_all() to extract all textual content from HTML tags,
        returning a single concatenated string.
    """
    html = fetch(url)
    text_from_page = parse_page_tags_all(html)
    content_page = ' '.join(text_from_page)
    return content_page 


def compute_fingerprint(text):
    """ 
    Input:
        text (str): Preprocessed or raw text content

    Output:
        int: 64-bit Simhash fingerprint value

    Description:
        The text is preprocessed into lemmatized tokens, then Simhash
        algorithm generates a fingerprint representing the content's features,
        useful for near-duplicate detection.    
    """
    words = preprocess(text)
    return Simhash(words).value


def hamming_distance(fp1, fp2):
    """ 
    Input:
        fp1 (int): First 64-bit fingerprint
        fp2 (int): Second 64-bit fingerprint

    Output:
        int: Number of differing bits (Hamming distance)

    Description:
        Computes the number of bit positions at which the two fingerprints differ,
        a measure of similarity between two content fingerprints.    
    """
    x = (fp1 ^ fp2) & ((1 << 64) - 1)
    distance = 0
    while x:
        distance += 1
        x &= x - 1
    return distance


def compute_age(lambda_, t):
    """ 
        Compute age value to decide page freshness.

    Input:
        lambda_ (float): Expected number of content changes per day
        t (float): Number of days since last crawl

    Output:
        float: Computed age metric for freshness decision

    Description:
        Calculates the age based on the formula considering the
        change frequency lambda and elapsed time t, used to estimate
        if a page should be re-fetched.
    """
    if lambda_ == 0:
        return t 
    return (t+lambda_*np.exp(-lambda_*t)-1)/lambda_


def calculate_page_type(content, url=""):    
    """
    Input:
        content (str): Lowercased textual content of the page
        url (str): URL of the page (optional)

    Output:
        str: One of 'frequent', 'occasional', 'static', 'default'

    Description:
        Applies heuristic rules based on URL substrings and keyword
        occurrences in content to classify the update frequency/type
        of the page, which affects crawling strategy.
    """
    content = content.lower()
    url = url.lower()
    # Special handling for news sites
    if "guardian" in url or "cnn.com" in url or "bbc.com" in url:
        if "live" in content and "update" in content:
            return "frequent"
    #   URL-based rules 
    if "wikipedia.org/wiki/" in url:
        return "static"
    if "live" in url or "breaking" in url:
        return "frequent"
    if any(k in url for k in ["calendar", "event", "workshop", "conference"]):
        return "occasional"
    if any(k in url for k in ["about", "privacy", "contact", "terms"]):
        return "static"
    # Content-based keyword count 
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
    Input:
        gaps (List[int]): List of gap-encoded positions (e.g. [4, 3, 8])

    Output:
        List[int]: List of absolute positions (e.g. [4, 7, 15])

    Description:
        Decodes a gap-encoded list of word positions to their original absolute positions.
    """
    if not gaps:
        return []
    positions = [gaps[0]]
    for gap in gaps[1:]:
        positions.append(positions[-1] + gap)
    return positions


def to_gap_encoding(positions):
    """
    Input:
        positions (List[int]): List of absolute positions (e.g. [4, 7, 15])

    Output:
        List[int]: List of gap-encoded positions (e.g. [4, 3, 8])

    Description:
        Encodes a list of absolute word positions as gaps between consecutive positions,
        to save space in the inverted index.
    """
    if not positions:
        return []
    return [positions[0]] + [positions[i] - positions[i-1] for i in range(1, len(positions))]



#   ----- Storage class -----

class Storage:

    """
    Class to manage persistence of pages and inverted index with freshness
    and near-duplicate detection.

    Attributes:
        pages_file (str): Path to JSON file storing page metadata
        index_file (str): Path to JSON file storing inverted index
        pages (dict): In-memory dictionary of page metadata
        inverted_index (dict): In-memory inverted index {term: {url: {tf, positions}}}
    """

    def __init__(self, pages_file="data/pages.json", index_file="data/inverted_index.json"):
        """
        Input:
            pages_file (str): Filename for storing pages metadata (default "data/pages.json")
            index_file (str): Filename for storing inverted index (default "data/inverted_index.json")

        Output:
            None

        Description:
            Initialize storage, loading pages and index from files or empty.
        """

        self.pages_file = pages_file
        self.index_file = index_file
        # Load data from disk or initialize empty
        self.pages = self._load_json(self.pages_file) or {}
        self.inverted_index = self._load_json(self.index_file) or {}


    def _load_json(self, filename):
        """
        Input:
            filename (str): Path to JSON file

        Output:
            dict or None: Parsed JSON data or None if file doesn't exist
        
        Description: 
            Load JSON data from file.
        """
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
        return None


    def _save_json(self, filename, data):
        """
        Input:
            filename (str): Path to JSON file
            data (dict): Data to save

        Output:
            None

        Description: 
            Save data as JSON to file.
        """
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


    def save_page(self, url, content):
        """
        Input:
            url (str): URL of the page
            content (str): Text content of the page

        Output:
            None

        Description:
            Computes page fingerprint and type, updates last fetch time,
            and updates inverted index with terms and their positions.
        """
        now = time.time()
        page_type = calculate_page_type(content, url)
        fingerprint = compute_fingerprint(content)

        print(f"Saving page: {url}")  # <-- DEBUG

        # Save/Update page data
        self.pages[url] = {
            "fingerprint": fingerprint,
            "page_type": page_type,
            "last_fetch": now 
        }

        # Update index terms automatically (not on the disk)
        self.index_terms(url, content)

    
    def index_terms(self, url, content):
        """
        Input:
            url (str): URL of the page
            content (str): Text content of the page

        Output:
            None

        Description:
            Preprocesses content to extract lemmatized words, calculates
            term frequencies and position lists, gap-encodes positions,
            and updates the inverted index accordingly.
        """
        # Ensure the URL is already present in the pages dictionary.
        if url not in self.pages:
            raise ValueError(f"URL {url} not found in pages")

        # Preprocess the content to extract clean, lemmatized words
        words = preprocess(content)

        tf = Counter()  # Term-frequency of the words 
        positions = {}  # Dictionary to store word positions in the document

        # Iterate through all words to build term frequencies and positions.
        for i, word in enumerate(words):
            tf[word] += 1
            positions.setdefault(word, []).append(i)

        # Perform position gap encoding:
        # For each word, store the list of position *gaps* instead of absolute positions
        for word in positions:
            positions[word] = to_gap_encoding(positions[word]) 

        # Remove existing entries in the inverted index for this URL
        # This avoids duplicating or mixing old and new data
        for term in list(self.inverted_index.keys()):
            if url in self.inverted_index[term]:
                del self.inverted_index[term][url]
                if not self.inverted_index[term]:
                    del self.inverted_index[term]

        # Add the new term frequency and position data into the inverted index
        for term in tf:
            if term not in self.inverted_index:
                self.inverted_index[term] = {}
            self.inverted_index[term][url] = {
                "tf": tf[term],
                "positions": positions[term]
            }

        # Save the updated inverted index to the index file (disk)
        self._save_json(self.index_file, self.inverted_index)


    def get_page_type(self, url):
        """
        Input:
            url (str): URL of the page

        Output:
            str or None: Page type string or None if page not found
        """
        page = self.pages.get(url)
        if page:
            return page.get("page_type")
        return None


    def get_last_fetch(self, url):
        """
        Input:
            url (str): URL of the page

        Output:
            float or None: UNIX timestamp of last fetch or None if not found
        """
        page = self.pages.get(url)
        if page:
            return page.get("last_fetch")
        return None


    def get_tf(self, term):
        """
        Input:
            term (str): Term to query

        Output:
            dict: Dictionary {url: {"tf": int, "positions": List[int]}} or empty dict
        """
        return self.inverted_index.get(term, {})


    def needs_refresh(self, url):
        """
        Input:
            url (str): URL of the page

        Output:
            bool: True if page needs fetching again, False otherwise

        Description:
            Calculates age of the page using compute_age with
            lambda frequency based on page type, and compares to threshold.
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
        Input:
            content (str): Text content of a page
            threshold (int): Maximum Hamming distance to consider near-duplicate (default 5)

        Output:
            Tuple[bool, str or None]: (True, url) if near-duplicate found, else (False, None)

        Description:
            Computes fingerprint of the new content and compares Hamming distance
            with all stored fingerprints to detect near duplicates. 
        """
        new_fp = compute_fingerprint(content)
    
        for url, page_data in self.pages.items():
            fp = page_data.get("fingerprint")
            if fp is None:
                continue
            try:
                d = hamming_distance(new_fp, fp)
                if d <= threshold:
                    return True, url  # near duplicate found, return url
            except Exception:
                continue

        return False, None  # the content is not a near-duplicate 
    

    def commit(self):
        """ 
        Input:
            None

        Output:
            None
        
        Description: 
            Saves the data on the disk 
        """
        self._save_json(self.pages_file, self.pages)
        self._save_json(self.index_file, self.inverted_index)


    def close(self):
        """
        Alias for commit to save data and close storage.

        Input:
            None

        Output:
            None
        """
        self.commit()
