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
import subprocess
import sys
import tempfile
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
    try:
        crawl_result = app.crawl(
            url,
            limit=50,
            scrape_options={"formats": ["markdown"]},
            poll_interval=5,
        )
        # crawl returns a list or object with .data
        raw_pages = crawl_result.data if hasattr(crawl_result, "data") else (crawl_result if isinstance(crawl_result, list) else [])
    except Exception as e:
        print(f"Firecrawl crawl failed ({e}), trying single-page scrape")
        # Fall back to single page scrape
        scrape_result = app.scrape(url, formats=["markdown"])
        raw_pages = [scrape_result]

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


# ── Scrapy in-process runner (used by subprocess) ────────────────────────────

def _run_scrapy_in_process(url: str, output_file: str) -> None:
    """Run Scrapy spider in the current process and write results to output_file."""
    settings = Settings()
    settings.set("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    process = CrawlerProcess(settings)

    collected_items = []

    def item_scraped(item, response, spider):
        collected_items.append(dict(item))

    dispatcher.connect(item_scraped, signal=signals.item_scraped)

    process.crawl(SiteSpider, start_url=url)
    process.start()

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

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nScraped {len(pages)} pages and saved to {output_file}")


SCRAPY_TIMEOUT_SECONDS = 300  # 5 minutes


# ── ScraperService (OOP wrapper) ─────────────────────────────────────────────

class ScraperService:
    """Unified web scraper: Scrapy primary, Firecrawl fallback."""

    @staticmethod
    def run_spider(url: str, output_file: str = "scraped_data.json") -> None:
        """
        Run SiteSpider on a URL with a 5-minute timeout.

        Falls back to Firecrawl if:
          - Scrapy times out (> 5 min)
          - Scrapy returns zero pages
          - Scrapy subprocess crashes
        """
        script = Path(__file__).resolve()
        cmd = [sys.executable, str(script), url, output_file, "--scrapy-only"]

        print(f"Starting Scrapy scrape for {url} (timeout: {SCRAPY_TIMEOUT_SECONDS}s)")

        try:
            result = subprocess.run(
                cmd,
                timeout=SCRAPY_TIMEOUT_SECONDS,
                capture_output=True,
                text=True,
            )
            scrapy_ok = False
            if result.returncode == 0 and Path(output_file).exists():
                with open(output_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("total_pages", 0) > 0:
                    scrapy_ok = True
                    print(f"\nScrapy scraped {data['total_pages']} pages and saved to {output_file}")

            if not scrapy_ok:
                print("\nNo items were scraped with Scrapy -- trying Firecrawl fallback")
                _run_firecrawl_scraper(url, output_file)

        except subprocess.TimeoutExpired:
            print(f"\nScrapy timed out after {SCRAPY_TIMEOUT_SECONDS}s -- falling back to Firecrawl")
            _run_firecrawl_scraper(url, output_file)

        except Exception as e:
            print(f"\nScrapy subprocess failed ({e}) -- falling back to Firecrawl")
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

    # --scrapy-only: run Scrapy in-process without fallback (used by subprocess timeout)
    if "--scrapy-only" in sys.argv:
        _run_scrapy_in_process(target_url, out_file)
    else:
        run_spider(target_url, out_file)
