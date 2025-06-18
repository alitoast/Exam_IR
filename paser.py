
"""
Web Parser Module
=================

Description:
------------
This module provides functionality to analyze the content of web pages,
extract relevant data (text, links), and clean or normalize it for further processing.
It is typically used in conjunction with a web crawler or fetcher module.

It includes:
- Normalization of extracted URLs using RFC 3986 standards
- Spider trap detection to avoid crawling problematic or infinite URLs
- Sitemap parsing (both XML and HTML)
- Extraction of links from page content (HTML parsing)
- Extraction of visible text content from HTML structure (e.g. `<p>`, `<h1>`, etc.)

Core Functions:
---------------

1. **normalize_url(url)**:
   - Normalizes a given URL using RFC 3986 (e.g., removes fragments, default ports, lowercases the host)
   - Returns a cleaned URL string

2. **check_spider_traps(url)**:
   - Applies heuristics to detect and block "spider traps" such as:
     - Excessively long URLs
     - Excessive path depth
     - Suspicious query parameters (e.g., session IDs, pagination loops)
   - Returns `False` if the URL is suspicious, otherwise `True`

3. **parse_sitemap(sitemap_list)**:
   - Parses a list of sitemap URLs (both XML and HTML)
   - Handles recursive sitemap indexes
   - Returns a DataFrame with URL, priority, and update frequency fields
   - Normalizes and filters URLs to avoid traps

4. **parse_page_url(html, sitemaps_urls, useragent)**:
   - Parses an HTML page and extracts all candidate URLs (from `href` and `src`)
   - Filters out:
     - URLs disallowed by `robots.txt`
     - Duplicates
     - Already known sitemap URLs
     - Potential spider traps
   - Returns a cleaned and filtered list of new URLs

5. **parse_page_tags_all(html, tags_type=['p', 'article', 'section', 'li', 'h1', 'h2', 'h3'])**:
   - Extracts visible textual content from an HTML page
   - Collects text from the specified tags (in DOM order)
   - Returns a list of clean, readable text blocks

Dependencies:
-------------
- `re`                 : Regular expressions for pattern matching (trap detection)
- `requests`           : HTTP requests (for sitemap retrieval)
- `urllib.parse`       : URL decomposition and analysis
- `xml.etree.ElementTree`: XML parsing for sitemap content
- `bs4` (BeautifulSoup): HTML parsing for link and content extraction
- `rfc3986`            : Standards-compliant URL normalization
- `pandas`             : For structured sitemap output

Use Cases:
----------
- Clean and validate link extraction for web crawling
- Prevent crawling of problematic or infinite-loop URLs
- Extract meaningful page content (for indexing or NLP)
- Integrate with fetcher modules for full scraping pipelines

Note:
-----
The `parse_sitemap()` function returns a DataFrame. You may want to save it or process it for link scheduling.
Ensure that `fetch()` (from the fetcher module) is available in the context when calling parsing functions that fetch content.

"""




def normalize_url(url):

  '''
    Input:
        url (str): The URL to be normalized.

    Output:
        str: The normalized URL as a string. If normalization fails, the original URL is returned.

    Description:
        Normalizes the input URL according to the RFC 3986 standard and returns a string representation.
        This function uses the `rfc3986` library to parse and normalize the URL according to the
        URI standard defined in RFC 3986. This includes handling issues such as case normalization,
        removing default ports, sorting query parameters (if applicable), and more.

        If an error occurs during normalization (e.g., invalid input), it catches the exception
        and returns the original URL as a fallback.
  '''

  try:
      uri = rfc3986.uri_reference(url).normalize()
      return uri.unsplit()

  except Exception as e:
      print(f"Errore nella normalizzazione URL: {url} – {e}")
      return url

def check_spider_traps(url):

  '''

    Input:
     url(str):The URL to be analyzed.

    Output:
      - Returns False if the URL is suspicious or considered a "spider trap."
      - Returns True if the URL seems safe to crawl.

    Description:
      Checks whether a given URL is potentially harmful or could trap a web crawler
      in infinite loops or unnecessary crawling paths.

  '''

  MAX_URL_LENGTH = 200   # Arbitrary maximum allowed URL length
  MAX_PATH_DEPTH = 6     # Maximum allowed number of slashes in path
  trap_pattern = re.compile(r"(calendar|sessionid|track|ref|sort|date=|page=\d{3,})", re.IGNORECASE)   # Pattern matching common signs of spider traps:
                                                                                                       # calendars, session IDs, tracking params, endless pagination, etc.

  link = urlparse(url)

  # Check URL length
  if len(link) > MAX_URL_LENGTH:
     return False

  # Check path depth (number of '/' in path)
  if link.path.count('/') > MAX_PATH_DEPTH:
     return False

  # Check for suspicious patterns in the URL
  if trap_pattern.search(url):
     return False

  return True

def parse_sitemap(sitemap_list):

    '''
      Input:
        sitemap_list (list): A list of sitemap URLs to parse.

      Output:
          Returns a DataFrame containing:
         - URL
         - Priority
         - Update frequency

      Description:
         Supports both XML and HTML sitemaps. Handles nested sitemap indexes.
         Returns a DataFrame with the parsed sitemap entries.

    '''
    namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    all_entries = []

    if not sitemap_list:
        print("Nessuna sitemap fornita.")
        return pd.DataFrame(columns=['url', 'priority', 'update'])

    def parse_single_sitemap(sitemap_url):
        try:
            response = requests.get(sitemap_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Errore durante il recupero di {sitemap_url}: {e}")
            return []

        content_type = response.headers.get("Content-Type", "").lower()

        # Sitemap XML
        if "xml" in content_type:
            try:
                root = ET.fromstring(response.content)
            except ET.ParseError as e:
                print(f"Errore nel parsing XML: {e}")
                return []

            # If it's a sitemap index, recursively parse child sitemaps
            if root.tag.endswith('index'):
                sitemap_urls = [loc.text for loc in root.findall('ns:sitemap/ns:loc', namespace)]
                entries = []
                for sub_url in sitemap_urls:
                    entries.extend(parse_single_sitemap(sub_url))
                return entries

            #  # If it's a standard URL set
            elif root.tag.endswith('urlset'):
                return [{
                    'url': url.findtext('ns:loc', default='', namespaces=namespace),
                    'priority': url.findtext('ns:priority', default=None, namespaces=namespace),
                    'update': url.findtext('ns:changefreq', default=None, namespaces=namespace)
                } for url in root.findall('ns:url', namespace)]

        # Handle HTML sitemaps (fallback)
        elif "html" in content_type:
            soup = BeautifulSoup(response.content, "html.parser")
            return [{
                'url': normalize_url(a['href']),
                'priority': None,
                'update': None
            } for a in soup.find_all('a', href=True)]

        else:
            print(f"Formato non riconosciuto per {sitemap_url} ({content_type})")
            return []

    #  Process each sitemap URL in the list
    for sitemap_url in sitemap_list:
        if sitemap_url:
            entries = parse_single_sitemap(sitemap_url)
            for entry in entries:
                url = normalize_url(entry['url']) #normalizzo l'url
                if check_spider_traps(url): # provo a limitare le spider-trap
                    entry['url'] = url

            all_entries.extend(entries)

    # Create a DataFrame and drop duplicates
    df = pd.DataFrame(all_entries, columns=['url', 'priority', 'update']).drop_duplicates()

    return df

def parse_page_url(html,sitemaps_urls,useragent=default_agent):

  '''
     Inputs:
     - html(str): The HTML content of the page.
     - sitemaps_urls(str): A list of URLs already known from sitemaps.
     - useragent(class): A user agent object containing disallowed paths (robots.txt rules).

     Output:
     - A list of new, allowed URLs extracted from the page.

     Description:
       Given an HTML page, this function extracts all URLs and filters them based on:
       - Exclusion of private or disallowed directories (from robots.txt rules)
       - Deduplication with URLs already found in the sitemap
       - Optional normalization and trap filtering

  '''

  # Parse HTML content
  soup = BeautifulSoup(html, 'html.parser')

  urls = []
  # Extract URLs from href attributes (e.g., <a>, <link>, etc.)
  for tag in soup.find_all(href=True):
      urls.append(tag['href'])

  # Extract URLs from src attributes (e.g., <img>, <script>, etc.)
  for tag in soup.find_all(src=True):
      urls.append(tag['src'])

  # Filter out disallowed URLs based on robots.txt rules
  if useragent.path_disallow != None:
    for url in urls:
      for path in useragent.path_disallow:
        if path in url:
         urls.remove(url)

  # Normalize URLs (e.g., remove fragments, resolve relative paths, etc.)
    urls = [normalize_url(url) for url in urls]

  # Remove duplicate URLs
    urls = list(set(urls))

  # Filter out potential spider traps
    urls = [url for url in urls if check_spider_traps(url)]

  # Filter out URLs that are already in the sitemap
  if sitemaps_urls != None:
    def_urls = sitemaps_urls
    for url in urls:
      if url not in def_urls:
         def_urls.append(url)
  else:
    def_urls = urls

  return def_urls


def parse_page_tags_all(html,tags_type = ['p','article','section','li','h1','h2','h3']):

  '''
    Inputs:
    - html: The HTML content as a string.
    - tags_type: A list of tag names to search for (default includes common content tags).

    Output:
    - A list of text strings extracted from the specified tags, preserving DOM order

    Description:
      Parses the given HTML content and extracts text from specified HTML tags.

  '''

  # Initialize the HTML parser
  soup = BeautifulSoup(html, "html.parser")

  # Find all tags of the specified types (respects DOM order)
  tags = soup.find_all(tags_type) #dovrebbe rispettare l'ordine del DOM

  # Extract clean text from each tag (removing whitespace and combining with spaces)
  texts = [tag.get_text(separator=' ', strip=True) for tag in tags]

  return texts

html = fetch(start_url_due, default_agent)
words = parse_page_tags_all(html)
print(words)
print(default_agent.last_access)


