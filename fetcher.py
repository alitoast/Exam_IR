"""
fetcher needs to be async-compatibile

TO DO:
vjeck the check_robots method if it's async compatibile
"""

import aiohttp

async def fetch(url, useragent=None):
    try:
        headers = {"User-Agent": useragent or default_agent.agent}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.text()
                return None
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None
