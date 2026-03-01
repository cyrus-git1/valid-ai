"""
src/helpers/playwright_scraper.py
---------------------------------
Playwright-based fallback scraper that intercepts API/XHR responses
to extract content from JavaScript-heavy sites and SPAs.

Only used when the primary Scrapy scraper returns no results.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, urljoin

import trafilatura
from playwright.async_api import async_playwright


MAX_PAGES = 20
NAV_TIMEOUT = 30_000  # 30s per page


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
        # Only capture same-domain or common API patterns
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
        # Fall back to domcontentloaded if networkidle times out
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
            await page.wait_for_timeout(3000)
        except Exception:
            page.remove_listener("response", _on_response)
            return None

    page.remove_listener("response", _on_response)

    # Extract rendered HTML content
    html = await page.content()
    page_text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
    title = await page.title()

    # Combine rendered text with intercepted API text
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
    domain = urlparse(start_url).netloc
    visited = set()
    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = await context.new_page()

        # Scrape the start URL
        result = await _scrape_page(page, start_url, domain)
        visited.add(start_url)
        if result:
            results.append(result)

        # Collect internal links from the rendered page
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

        # Visit internal links up to MAX_PAGES
        for link in internal_links[: MAX_PAGES - 1]:
            result = await _scrape_page(page, link, domain)
            if result:
                results.append(result)

        await browser.close()

    return results


def run_playwright_scraper(url, output_file="scraped_data.json"):
    """Sync entry point — runs the async Playwright scraper and writes JSON."""
    print(f"\n⚙ Falling back to Playwright scraper for {url}")
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

    print(f"✓ Playwright scraped {len(pages)} pages and saved to {output_path}")


if __name__ == "__main__":
    target_url = sys.argv[1] if len(sys.argv) > 1 else "https://www.torontomotors.ca/"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "scraped_data.json"
    run_playwright_scraper(target_url, out_file)
