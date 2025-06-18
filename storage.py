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


#   Functions imported from other modules of the project
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
    """ Map POS tags from Treebank to Wordnet """
    if tag.startswith('J'): return ADJ
    if tag.startswith('V'): return VERB
    if tag.startswith('N'): return NOUN
    if tag.startswith('R'): return ADV
    return NOUN


def preprocess(text):
    """
    Preprocesses the page content:
    - Lowercases and tokenizes text
    - Removes stop words, numbers and punctuation
    - Lemmatizes each word using POS tags
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
    """ Given the url of a web page, returns the content of the page as a string """
    html = fetch(url)
    text_from_page = parse_page_tags_all(html)
    content_page = ' '.join(text_from_page)
    return content_page 


def compute_fingerprint(text):
    """ Given a text, computes its Simhash fingerprint. """
    words = preprocess(text)
    return Simhash(words).value


def hamming_distance(fp1, fp2):
    """ 
    Given two fingerprints, computes the hamming distance between. 
    """
    x = (fp1 ^ fp2) & ((1 << 64) - 1)
    distance = 0
    while x:
        distance += 1
        x &= x - 1
    return distance


def compute_age(lambda_, t):
    """ 
    Computes the age, given: 
    -   Lambda: number of times per day the page is changed 
    -   t: days since the last crawl 
    """
    if lambda_ == 0:
        return t 
    return (t+lambda_*np.exp(-lambda_*t)-1)/lambda_


def calculate_page_type(content, url=""):    
    """
    Classifies the page based on the content and the url, returns one between: 'frequent', 
    'occasional', 'static', 'default'
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
    Given a list of gap-encoded positions, returns the absolute positions. 
    Ex: [4,3,8] → [4, 7, 15]
    """
    if not gaps:
        return []
    positions = [gaps[0]]
    for gap in gaps[1:]:
        positions.append(positions[-1] + gap)
    return positions


def to_gap_encoding(positions):
    """
    Given a list of absolute positions, it returns the gap-encoded positions.
    Ex: [4, 7, 15] → [4, 3, 8]
    """
    if not positions:
        return []
    return [positions[0]] + [positions[i] - positions[i-1] for i in range(1, len(positions))]



#   ----- Storage class -----

class Storage:


    def __init__(self, pages_file="data/pages.json", index_file="data/inverted_index.json"):
        self.pages_file = pages_file
        self.index_file = index_file
        # Load data from disk or initialize empty
        self.pages = self._load_json(self.pages_file) or {}
        self.inverted_index = self._load_json(self.index_file) or {}


    def _load_json(self, filename):
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
        return None


    def _save_json(self, filename, data):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


    def save_page(self, url, content, fingerprint):
        now = time.time()
        page_type = calculate_page_type(content, url)

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
        Indexes the terms contained in a web page with a certain url and content. 
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
        """
        Checks for freshness using age.  
        States if a page needs to be fetched again by defining lambda dynamically by looking at the "type
        of page". 
        In fact, different pages are being updated at different frequency depending on the type of content:
        a news page is probably updated multiple times per day, while the contacts page of a company is 
        probably updated once a month. 
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
        Checks whether the content of a web page is a near-duplicate of a web page 
        in the set. 
        This is done by computing the fingerprint of the content of the page using the Simhash algorithm
        and then computing the hamming distance with the fingerprints of the pages contained in the set 
        pages. 
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
        """ Saves the data on the disk """
        self._save_json(self.pages_file, self.pages)
        self._save_json(self.index_file, self.inverted_index)


    def close(self):
        self.commit()
