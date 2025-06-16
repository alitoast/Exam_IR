"""
Scheduler expects parser.extract_links(url, html) — but in parser.py-eva we have: parse_page_url(html, sitemap_urls, useragent=...)

so adding an adapter method for it
"""

class ParserWrapper:

    def __init__(self, sitemap_urls=None, useragent=None):
        self.sitemap_urls = sitemap_urls
        self.useragent = useragent or default_agent

    def extract_links(self, url, html):
        return parse_page_url(html, self.sitemap_urls, self.useragent)