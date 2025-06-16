#   Implementare database
#   Implementare funzione per near-duplicates 
#   Implementare indici 
#   Controllo freshness

"""
SQLite is a C library that provides a lightweight disk-based database that doesn’t require 
a separate server process and allows accessing the database using a nonstandard variant of the 
SQL query language.
I import it and use it to save and manage the data.
"""

import sqlite3  # library to import      
import time 
import nltk # library to import 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize 
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV

from simhash import Simhash
from collections import Counter 

#   Functions imported from other modules 
from fetcher import fetch 
from parser import parse_page_tags

nltk.download('stopwords')
nltk.download('wordnet') 
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')  
nltk.download('omw-1.4')


#   Functions that will be useful 

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
#   Da completare  

def content_page(url): 
    html = fetch(url)


def preprocess(text): 

    words = word_tokenize(text.lower())
    
    #   Set of stopwords 
    stop_words = set(stopwords.words('english')) 
    
    #   Lemmatizer 
    lemmatizer = WordNetLemmatizer()

    number_words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']

    #   Remove stop words, number_words and punctuation 
    words = [w for w in words if w.isalpha() and w not in stop_words and w not in number_words]

    tagged_words = pos_tag(words)

    #   Lemmatization is done using the correct grammatical type 
    lemmatized_words = [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged_words]

    return lemmatized_words


def compute_fingerprint(text):
    
    words = preprocess(text)
    return Simhash(words).value


def hamming_distance(fp1, fp2): 
    x = (fp1 ^ fp2) & ((1 << 64) - 1)
    distance = 0
    while x:
        distance += 1
        x &= x - 1
    return distance



class StorageDB : 

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
            fingerprint TEXT,
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


    def save_page(self, url, content, fingerprint):    
        
        now = time.time() 
        self.conn.execute('''
        INSERT OR REPLACE INTO pages (url, content, fingerprint, last_fetch)
        VALUES(?, ?, ?, ?)
        ''', (url, content, fingerprint, now))
        self.conn.commit() 


    def get_last_fetch(self, url): 

        cursor = self.conn.cursor() 
        cursor.execute('SELECT last_fetch FROM pages WHERE url = ?', (url,))
        result = cursor.fetchone() 
        return result[0] if result else None


    #   Not sure this function is necessary    
    def get_fingerprint(self, url): 

        cursor = self.conn.cursor() 
        
        cursor.execute('''
        SELECT fingerprint FROM pages WHERE url =?
        ''', (url,))

        result = cursor.fetchone()
        return result[0] if result else None 


    ''' Given an url, which was saved in the table pages, index_terms extract the important terms
        that are present in the document, it calculates tf and it adds them in the inverted index.
        First, I preprocess the content of the page, in order to index only the "important" terms. 
        There are a few steps:
        -   all the letters are converted to lower  
        -   togliere stop words
        -   stemming/lemmatization 
    '''

    #   This method can be called only after the "page" and its characteristics have been saved
    #   in the table pages 

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


    """
    
    """

    def is_near_duplicate(self, content, threshold=5):
        
        new_fp = compute_fingerprint(content) 
        cursor = self.conn.cursor() 
        cursor.execute('SELECT url, fingerprint WHERE fingerprint IS NOT NULL')
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
    Check for freshness: is it better to use freshness or age? 
    I think age is better. 

    3 ways to implement it: 
    1.  Constant value of lambda 
    2.  Value that depends on how often the page changes. For this we need to have a history of the 
        changes --> necessary to use another table page_changes
    3.  Value that depends on page type or domain --> how to implement it 

    I will start with way 1 with lambda_ = 0.01

    code to add in scheduler.py 

    from storage import Storage

    storage = Storage() 
    
    def should_refresh(self, url):    

           
    """

    """
    Funzioni che non ho ancora implementato 

    def needs_refresh(): 

    def close(self):
        self.conn.close() 

"""