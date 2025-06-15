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
import re   
import nltk # library to import 
from nltk import stopwords 

class StorageDB : 

    def __init__(self, db_path="/data/storage.db"):
        self.conn = sqlite3.connect(db_path)
        cursor = self.conn.cursor() 
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

    def get_fingerprint(self, url): 
        cursor = self.conn.cursor() 
        cursor.execute('''
        SELECT fingerprint FROM pages WHERE url =?
        ''', (url,))
        result = cursor.fetchone()
        return result[0] if result else None 

    ''' Given an url, which was saved in the table pages, this method extract the important terms
        that are present in the document, it calculates tf and it adds them in the inverted index.
        There are a few steps:
        -   all the letters are converted to lower  
        -   togliere stop words
        -   stemming/lemmatization 
    '''
    def index_terms(self, url):
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, content FROM pages WHERE url=?', (url,))
        result = cursor.fetchone() 
        if not result:
            raise ValueError(f"URL {url} non trovato in pages")
        page_id, content = result

        words = re.findall(r'\w+', content.lower())
    

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

    def compute_fingerprint(self, content):

    def index_terms(self, url, text): 

    def needs_refresh(): 


    def is_near_duplicate_fingerprint():

    def close(self):
        self.conn.close() 

"""