import scrapy
from urllib.parse import urlparse
import trafilatura

class SiteSpider(scrapy.Spider):
    name = "site"
    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 1.0,
        "AUTOTHROTTLE_MAX_DELAY": 10.0,
        "DOWNLOAD_TIMEOUT": 20,
        "RETRY_TIMES": 2,
        "CONCURRENT_REQUESTS": 8,
        "LOG_LEVEL": "INFO",
    }

    def __init__(self, start_url: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = [start_url]
        self.allowed_domains = [urlparse(start_url).netloc]

    def parse(self, response):
        html = response.text
        text = trafilatura.extract(html, include_comments=False, include_tables=False)
        if text:
            yield {
                "url": response.url,
                "title": response.css("title::text").get(),
                "text": text,
            }

        # follow internal links
        for href in response.css("a::attr(href)").getall():
            if href and not href.startswith(('tel:', 'mailto:', 'javascript:', '#', 'data:')):
                try:
                    yield response.follow(href, callback=self.parse)
                except ValueError:
                    continue