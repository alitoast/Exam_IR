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

import sqlite3     
import time 

class StorageDB : 

    def __init__(self, db_path="storage.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables() 

    """
    Function to create the two tables we need: pages and inverted_index
    -   pages:
        -   url: url of the page (obvious)
        -   html: content of the page in html 
        -   text: onlt the text of the page, so the html content is "filtered" 
        -   fingerprint: hashed representation of the content of the page 
        -   last_fetch: date in which the document was last fetched 
    -   inverted_index:
        -   term: word extracted from the context of a web page
        -   url: url of the page (obvious)
    """
    def _create_tables(self): 
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS pages(
            url TEXT PRIMARY KEY,
            html TEXT,
            text TEXT, 
            fingerprint TEXT,
            last_fetch TIMESTAMP 
        )
        ''')

        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS inverted_index(
            term TEXT, 
            url TEXT, 
            PRIMARY KEY (term, url), 
            FOREIGN KEY (url) REFERENCES pages(url) ON DELETE CASCADE
        )
        ''')

        self.conn.commit()

    def save_page(self, url, html, text, fingerprint, last_fetch):    
        now = time.time() 
        self.conn.execute('''
        INSERT OR INSERT INTO pages (url, html, text, fingerprint, last_fetch)
        VALUES(?, ?, ?, ?, ?, ?)
        ''', (url, html, text, fingerprint, now))
        self.conn.commit() 


    """
    Funzioni che non ho ancora implementato 
    def index_terms(self, url, text): 


    def compute_fingerprint(): 


    def needs_refresh(): 


    def is_near_duplicate_fingerprint():

    def close(self):
        self.conn.close() 

"""