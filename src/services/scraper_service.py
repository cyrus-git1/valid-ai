"""
src/services/scraper_service.py
--------------------------------
Web scraping service with dual strategy: Scrapy primary, Firecrawl fallback.

Also serves as the CLI entry point for subprocess invocation:
    python -m src.services.scraper_service <url> [output_file.json]
    python src/services/scraper_service.py <url> [output_file.json]
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import dotenv
import scrapy
import trafilatura
from scrapy.crawler import CrawlerProcess
from scrapy.settings import Settings
from scrapy import signals
from scrapy.signalmanager import dispatcher

dotenv.load_dotenv()


# ── Scrapy Spider ─────────────────────────────────────────────────────────────

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

        for href in response.css("a::attr(href)").getall():
            if href and not href.startswith(('tel:', 'mailto:', 'javascript:', '#', 'data:')):
                try:
                    yield response.follow(href, callback=self.parse)
                except ValueError:
                    continue


# ── Firecrawl fallback ───────────────────────────────────────────────────────

def _run_firecrawl_scraper(url: str, output_file: str = "scraped_data.json") -> None:
    """Crawl a site using the Firecrawl API and save results to JSON."""
    from firecrawl import FirecrawlApp

    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FIRECRAWL_API_KEY is not set. Add it to .env to use the Firecrawl fallback."
        )

    print(f"\nFalling back to Firecrawl for {url}")

    app = FirecrawlApp(api_key=api_key)

    # Use crawl for full site crawl (follows internal links)
    crawl_result = app.crawl_url(
        url,
        params={
            "limit": 50,
            "scrapeOptions": {
                "formats": ["markdown"],
            },
        },
        poll_interval=5,
    )

    # crawl_result is a CrawlStatusResponse with .data list
    raw_pages = crawl_result.data if hasattr(crawl_result, "data") else []

    pages = []
    for idx, item in enumerate(raw_pages, start=1):
        # Each item has .markdown, .metadata (with .title, .sourceURL, etc.)
        markdown = item.markdown if hasattr(item, "markdown") else (item.get("markdown") if isinstance(item, dict) else "")
        metadata = item.metadata if hasattr(item, "metadata") else (item.get("metadata", {}) if isinstance(item, dict) else {})

        if isinstance(metadata, dict):
            page_title = metadata.get("title", "")
            page_url = metadata.get("sourceURL", metadata.get("url", ""))
        else:
            page_title = getattr(metadata, "title", "")
            page_url = getattr(metadata, "sourceURL", getattr(metadata, "url", ""))

        text_parts = []
        if page_title:
            text_parts.append(f"Title: {page_title}")
        if page_url:
            text_parts.append(f"URL: {page_url}")
        if markdown:
            text_parts.append(markdown)

        full_text = "\n\n".join(text_parts)
        if full_text.strip():
            pages.append({
                "page": idx,
                "url": page_url or "",
                "title": page_title or "",
                "text": full_text,
            })

    output_data = {
        "source_url": url,
        "scraped_at": datetime.now().isoformat(),
        "total_pages": len(pages),
        "pages": pages,
    }

    output_path = Path(output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Firecrawl scraped {len(pages)} pages and saved to {output_path}")


# ── ScraperService (OOP wrapper) ─────────────────────────────────────────────

class ScraperService:
    """Unified web scraper: Scrapy first, Firecrawl fallback."""

    @staticmethod
    def run_spider(url: str, output_file: str = "scraped_data.json") -> None:
        """Run SiteSpider on a URL and save results to a JSON file."""
        settings = Settings()
        settings.set("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

        process = CrawlerProcess(settings)

        collected_items = []

        def item_scraped(item, response, spider):
            collected_items.append(dict(item))

        dispatcher.connect(item_scraped, signal=signals.item_scraped)

        process.crawl(SiteSpider, start_url=url)
        process.start()

        if collected_items:
            pages = []
            for idx, item in enumerate(collected_items, start=1):
                text_parts = []
                if item.get("title"):
                    text_parts.append(f"Title: {item['title']}")
                if item.get("url"):
                    text_parts.append(f"URL: {item['url']}")
                if item.get("text"):
                    text_parts.append(item["text"])

                full_text = "\n\n".join(text_parts)
                if full_text.strip():
                    pages.append({
                        "page": idx,
                        "url": item.get("url", ""),
                        "title": item.get("title", ""),
                        "text": full_text,
                    })

            output_data = {
                "source_url": url,
                "scraped_at": datetime.now().isoformat(),
                "total_pages": len(pages),
                "pages": pages,
            }

            output_path = Path(output_file)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"\nScraped {len(pages)} pages and saved to {output_path}")
            print(f"  File is ready for tokenization")
        else:
            print("\nNo items were scraped with Scrapy -- trying Firecrawl fallback")
            _run_firecrawl_scraper(url, output_file)


# Module-level convenience for backward compat
def run_spider(url: str, output_file: str = "scraped_data.json") -> None:
    ScraperService.run_spider(url, output_file)


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Add project root to path so imports resolve when run as subprocess
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    target_url = sys.argv[1] if len(sys.argv) > 1 else "https://www.torontomotors.ca/"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "scraped_data.json"
    run_spider(target_url, out_file)
