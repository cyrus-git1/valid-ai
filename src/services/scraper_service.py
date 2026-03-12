"""
src/services/scraper_service.py
--------------------------------
Web scraping service with dual strategy: Scrapy primary, Playwright fallback.

Also serves as the CLI entry point for subprocess invocation:
    python -m src.services.scraper_service <url> [output_file.json]
    python src/services/scraper_service.py <url> [output_file.json]
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, urljoin

import scrapy
import trafilatura
from scrapy.crawler import CrawlerProcess
from scrapy.settings import Settings
from scrapy import signals
from scrapy.signalmanager import dispatcher


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


# ── Playwright helpers ────────────────────────────────────────────────────────

MAX_PAGES = 20
NAV_TIMEOUT = 30_000


def _extract_strings(obj, min_length=30):
    """Recursively extract meaningful string values from a JSON object."""
    strings = []
    if isinstance(obj, str):
        stripped = obj.strip()
        if len(stripped) >= min_length:
            strings.append(stripped)
    elif isinstance(obj, dict):
        for v in obj.values():
            strings.extend(_extract_strings(v, min_length))
    elif isinstance(obj, list):
        for item in obj:
            strings.extend(_extract_strings(item, min_length))
    return strings


async def _scrape_page(page, url, domain):
    """Navigate to a single URL, intercept API responses, and extract text."""
    api_texts = []

    async def _on_response(response):
        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            return
        resp_domain = urlparse(response.url).netloc
        if domain not in resp_domain and "api" not in response.url.lower():
            return
        try:
            body = await response.json()
            strings = _extract_strings(body)
            api_texts.extend(strings)
        except Exception:
            pass

    page.on("response", _on_response)

    try:
        await page.goto(url, wait_until="networkidle", timeout=NAV_TIMEOUT)
    except Exception:
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
            await page.wait_for_timeout(3000)
        except Exception:
            page.remove_listener("response", _on_response)
            return None

    page.remove_listener("response", _on_response)

    html = await page.content()
    page_text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
    title = await page.title()

    api_content = "\n\n".join(api_texts)
    if api_content and api_content not in page_text:
        combined = f"{page_text}\n\n{api_content}" if page_text else api_content
    else:
        combined = page_text

    if not combined.strip():
        return None

    return {"url": url, "title": title, "text": combined.strip()}


async def _scrape_with_playwright(start_url):
    """Scrape a site using Playwright with API response interception."""
    from playwright.async_api import async_playwright

    domain = urlparse(start_url).netloc
    visited = set()
    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = await context.new_page()

        result = await _scrape_page(page, start_url, domain)
        visited.add(start_url)
        if result:
            results.append(result)

        links = await page.eval_on_selector_all(
            "a[href]",
            "els => els.map(e => e.href)"
        )

        internal_links = []
        for href in links:
            if not href or href.startswith(("tel:", "mailto:", "javascript:", "#", "data:")):
                continue
            resolved = urljoin(start_url, href)
            parsed = urlparse(resolved)
            normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if parsed.netloc == domain and normalized not in visited:
                visited.add(normalized)
                internal_links.append(normalized)

        for link in internal_links[: MAX_PAGES - 1]:
            result = await _scrape_page(page, link, domain)
            if result:
                results.append(result)

        await browser.close()

    return results


def _run_playwright_scraper(url, output_file="scraped_data.json"):
    """Sync entry point for the Playwright fallback scraper."""
    print(f"\nFalling back to Playwright scraper for {url}")

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    results = asyncio.run(_scrape_with_playwright(url))

    pages = []
    for idx, item in enumerate(results, start=1):
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

    print(f"Playwright scraped {len(pages)} pages and saved to {output_path}")


# ── ScraperService (OOP wrapper) ─────────────────────────────────────────────

class ScraperService:
    """Unified web scraper: Scrapy first, Playwright fallback."""

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
            print("\nNo items were scraped with Scrapy -- trying Playwright fallback")
            _run_playwright_scraper(url, output_file)


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
