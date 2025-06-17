"""
SQLite is a C library that provides a lightweight disk-based database that doesn’t require 
a separate server process and allows accessing the database using a nonstandard variant of the 
SQL query language. It is imported and used it to save and manage data. 
Time is necessary to manage time. lol
Nltk (Naturale Language Toolkit) is a suite of libraries and programs for symbolic and 
statistical natural language processing (NLP) for English. From this suite, several functions were
imported.   
"""
import sqlite3                      #   library to import      
import time 
import nltk                         #   library to import 
import numpy as np
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize 
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV

"""
From the simhash library I import the Simhash object to apply the algorithm.
From the collections library I import the Counter object. 
"""
from simhash import Simhash         #   library to import 
from collections import Counter     #   library to import 


#   Functions imported from other modules 
from fetcher import fetch 
from parser import parse_page_tags_all 


#   Download necessary NLTK resources 
nltk.download('stopwords')
nltk.download('wordnet') 
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')  
nltk.download('omw-1.4')



LAMBDA_BY_TYPE = {
    "frequent": 2.0, 
    "occasional": 1.0,
    "static": 0.05,
    "default": 0.5 
}


#   Function to map POS tags from Treebank to Wordnet    

def get_wordnet_pos(treebank_tag): 

    if treebank_tag.startswith('J'):
        return ADJ

    elif treebank_tag.startswith('V'):
        return VERB

    elif treebank_tag.startswith('N'):
        return NOUN

    elif treebank_tag.startswith('R'):
        return ADV

    else:
        return NOUN  # default


#   Given the url, it returns the content of the page as a string.

def content_page(url): 

    html = fetch(url)
    text_from_page = parse_page_tags_all(html)
    content_page = ' '.join(text_from_page)

    return content_page 
    

#   Function to preprocess the content of a page before indexing the terms contained in it 

def preprocess(text): 

    words = word_tokenize(text.lower())
    
    #   Set of stopwords 
    stop_words = set(stopwords.words('english')) 
    
    #   Lemmatizer 
    lemmatizer = WordNetLemmatizer()

    #   These words are not in the stop words set, but I do not want them in the index, so they 
    #   will be removed 
    number_words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']

    #   Remove stop words, number_words and punctuation 
    words = [w for w in words if w.isalpha() and w not in stop_words and w not in number_words]

    #   Each words is coupled with 
    tagged_words = pos_tag(words)

    #   Lemmatization is done using the correct grammatical type 
    lemmatized_words = [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged_words]

    return lemmatized_words


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


#   Function that computes the age, given: 
#   -   Lambda: number of times per day the page is changed 
#   -   t: days since the last crawl 

def compute_age(lambda_, t): 

    return (t+lambda_*np.exp(-lambda_*t)-1)/lambda_


#   Function that etermines the type of the page based on its content keywords

def calculate_page_type(content):

    content = content.lower() 

    frequent_keywords = ["breaking news", "live updates", "report", "news"]         #   lambda = 2 
    occasional_keywords = ["calendar", "workshop", "conference", "event"]           #   lambda = 1
    static_keywords = ["contacts", "about us", "company info", "privacy policy"]    #   lambda = 0.05 

    #   for default I use lambda = 0.5 

    is_frequent = any(word in content for word in frequent_keywords)
    is_occasional = any(word in content for word in occasional_keywords)
    is_static = any(word in content for word in static_keywords)

    if is_frequent: 
        return "frequent"
    elif is_occasional: 
        return "occasional"
    elif is_static:
        return "static"
    else:
        return "default" 



#   Class that manages the SQLite storage, including pages and inverted index


class Storage : 

    def __init__(self, db_path="/data/storage.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables() 

    """
    Function to create the two tables we need: pages and inverted_index
    -   pages:
        -   id: si autoincrementa 
        -   url: url of the page 
        -   html: content of the page in html (maybe I can eliminate it)
        -   content: only the text of the page, so the html content is "filtered" 
        -   fingerprint: hashed representation of the content of the page 
        -   last_fetch: date in which the document was last fetched 
    -   inverted_index:
        -   term: word extracted from the context of a web page
        -   url: url of the page (obvious)
    """

    def _create_tables(self): 
        
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS pages(
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            url TEXT,
            content TEXT, 
            fingerprint INTEGER,
            page_type TEXT, 
            last_fetch TIMESTAMP 
        )
        ''')

        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS inverted_index(
            term TEXT, 
            page_id TEXT, 
            tf INTEGER,
            PRIMARY KEY (term, page_id), 
            FOREIGN KEY (page_id) REFERENCES pages(id) ON DELETE CASCADE
        )
        ''')

        self.conn.commit()
    
    
    #   Save or update a page on the database 
    def save_page(self, url, content, fingerprint):    
        
        now = time.time() 
        type = calculate_page_type(content) 

        self.conn.execute('''
        INSERT OR REPLACE INTO pages (url, content, fingerprint, page_type, last_fetch)
        VALUES(?, ?, ?, ?)
        ''', (url, content, fingerprint, type, now))
        
        self.conn.commit() 


    #   Retrieve the type of the page with a given url 
    def get_page_type(self, url): 

        cursor = self.conn.cursor() 
        cursor.execute('SELECT page_type FROM pages WHERE url = ?', (url,))
        result = cursor.fetchone() 
        return result[0] if result else None


    #   Retrieve the time where a page with a given url was last fetched 
    def get_last_fetch(self, url): 

        cursor = self.conn.cursor() 
        cursor.execute('SELECT last_fetch FROM pages WHERE url = ?', (url,))
        result = cursor.fetchone() 
        return result[0] if result else None


    #   Retrieve the fingerprint of a page with a given url     
    def get_fingerprint(self, url): 

        cursor = self.conn.cursor() 
        
        cursor.execute('''
        SELECT fingerprint FROM pages WHERE url =?
        ''', (url,))

        result = cursor.fetchone()
        return result[0] if result else None 


    #   Given an url, which was saved in the table pages, index_terms extract the important terms
    #   that are present in the document, it calculates tf and it adds them in the inverted index.

    def index_terms(self, url): 
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, content FROM pages WHERE url=?', (url,))
        result = cursor.fetchone() 
        if not result:
            raise ValueError(f"URL {url} non trovato in pages")
        page_id, content = result

        #   Preprocess the content of the page 
        words = preprocess(content)

        #   Count how many times the word 
        tf = Counter(words) 

        # Clean the old terms (is it necessary?)
        cursor.execute('DELETE FROM inverted_index WHERE page_id = ?', (page_id,))
    
        for term, freq in tf.items():
            cursor.execute('''
                INSERT OR REPLACE INTO inverted_index (term, page_id, tf)
                VALUES (?, ?, ?)
            ''', (term, page_id, freq))

        self.conn.commit()


    #   It checks whether a content is the near-duplicate of a content saved in the pages table
    def is_near_duplicate(self, content, threshold=5):
        
        new_fp = compute_fingerprint(content) 
        cursor = self.conn.cursor() 
        cursor.execute('SELECT url, fingerprint FROM pages WHERE fingerprint IS NOT NULL')
        rows = cursor.fetchall()

        for url, fp in rows:
            try: 
                #   Calculation of the Hamming distance between the fingerprints, in order 
                #   to check if they are near duplicates 
                d = hamming_distance(new_fp, fp)
                if d <= threshold :
                    return True     #   The pages are near-duplicates, we can also return the url if we want
            except ValueError:
                continue    #   These two pages are not near-duplicates, the algorithm can proceed with 
                            #   the next one 
        
        return False    #   This page is not a near-duplicate of any of the pages saved in the db 
    

    """
    Check for freshness: I used age for determining the freshness or not of a page. 

    This function will state if a page needs to be fetched again by defining lambda dynamically by 
    looking at the "type of page". 
    In fact, different pages are being updated at different frequency depending on the type of content:
    a news page is probably updated multiple times per day, while the contacts page of a company is 
    probably updated once a month. 
    
    """
    def needs_refresh(self, url):   
        last_fetch = self.get_last_fetch(url)
        
        if last_fetch is None: 
            return True

        page_type = self.get_page_type(url)
        now = time.time() 

        time_in_days = (now - last_fetch)/86400 

        lambda_ = LAMBDA_BY_TYPE.get(page_type)

        threshold = 1 / lambda_

        age = compute_age(lambda_, time_in_days)

        return age > threshold 


    #   Close the database connection    
    def close(self):
        self.conn.close() 
