#   Libraries to import 
import json
import time
import os
import numpy as np
from collections import Counter
from simhash import Simhash 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV

#   Functions imported from other modules of the project
from fetcher import fetch 
from parser import parse_page_tags_all 

#   Download necessary NLTK resources 
nltk.download('stopwords')
nltk.download('wordnet') 
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')  
nltk.download('omw-1.4')

#   Function to map POS tags from Treebank to Wordnet    
def get_wordnet_pos(tag):
    if tag.startswith('J'): return ADJ
    if tag.startswith('V'): return VERB
    if tag.startswith('N'): return NOUN
    if tag.startswith('R'): return ADV
    return NOUN

#   Function to preprocess the content of a page before indexing the terms contained in it 
def preprocess(text):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    #   These words are not in the stop words set, but I do not want them in the index, so they 
    #   will be removed 
    number_words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
    words = word_tokenize(text.lower())
    #   Remove stop words, number_words and punctuation 
    words = [w for w in words if w.isalpha() and w not in stop_words and w not in number_words]
    #   Each words is coupled with their tag 
    tagged_words = pos_tag(words)    
    #   Lemmatization is done using the right grammatical type and then the words are returned 
    return [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged_words]

#   Given the url, it returns the content of the page as a string.
def content_page(url): 
    html = fetch(url)
    text_from_page = parse_page_tags_all(html)
    content_page = ' '.join(text_from_page)
    return content_page 

#   Given the text, this function computes its Simhash fingerprint 
def compute_fingerprint(text):
    words = preprocess(text)
    return Simhash(words).value

#   Given two fingerprint, this function computes the hamming distance between them.
#   It will be used to check if two texts are near-duplicates or not 
def hamming_distance(fp1, fp2):
    x = (fp1 ^ fp2) & ((1 << 64) - 1)
    distance = 0
    while x:
        distance += 1
        x &= x - 1
    return distance

#   Lambda mapping for different page types 
LAMBDA_BY_TYPE = {
    "frequent": 2.0,
    "occasional": 1.0,
    "static": 0.05,
    "default": 0.5
}

#   Function that computes the age, given: 
#   -   Lambda: number of times per day the page is changed 
#   -   t: days since the last crawl 
def compute_age(lambda_, t):
    return (t+lambda_*np.exp(-lambda_*t)-1)/lambda_

#   Function that etermines the type of the page based on its content keywords
def calculate_page_type(content):
    content = content.lower()
    frequent_keywords = ["breaking news", "live updates", "report", "news"]
    occasional_keywords = ["calendar", "workshop", "conference", "event"]
    static_keywords = ["contacts", "about us", "company info", "privacy policy"]

    if any(k in content for k in frequent_keywords):
        return "frequent"
    if any(k in content for k in occasional_keywords):
        return "occasional"
    if any(k in content for k in static_keywords):
        return "static"
    return "default"



class Storage:

    def __init__(self, pages_file="pages.json", index_file="inverted_index.json"):
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
        page_type = calculate_page_type(content)

        # Save/Update page data
        self.pages[url] = {
            "content": content,
            "fingerprint": fingerprint,
            "page_type": page_type,
            "last_fetch": now
        }
        self._save_json(self.pages_file, self.pages)

        # Update index terms automatically
        self.index_terms(url)

    def index_terms(self, url):
        if url not in self.pages:
            raise ValueError(f"URL {url} not found in pages")

        content = self.pages[url]["content"]
        words = preprocess(content)
        tf = Counter(words)

        # Remove old terms for this url from inverted index
        for term in list(self.inverted_index.keys()):
            if url in self.inverted_index[term]:
                del self.inverted_index[term][url]
                # Clean empty dict
                if not self.inverted_index[term]:
                    del self.inverted_index[term]

        # Insert new tf for this url
        for term, freq in tf.items():
            if term not in self.inverted_index:
                self.inverted_index[term] = {}
            self.inverted_index[term][url] = freq

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
        new_fp = self.compute_fingerprint(content)
    
        for url, page_data in self.pages.items():
            fp = page_data.get("fingerprint")
            if fp is None:
                continue
            try:
                d = self.hamming_distance(new_fp, int(fp))
                if d <= threshold:
                    return True, url  # near duplicate found, optionally return url
            except Exception:
                continue
        return False, None
