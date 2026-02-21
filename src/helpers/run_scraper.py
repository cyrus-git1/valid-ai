"""
src/helpers/run_scraper.py
----------------------------
CLI scraper runner — runs SiteSpider and saves output as tokenization-ready JSON.

Called as a subprocess by IngestService._run_spider_subprocess() to avoid
Scrapy's one-CrawlerProcess-per-process limitation inside the FastAPI server.

Usage
-----
    python -m src.helpers.run_scraper <url> [output_file.json]
    python src/helpers/run_scraper.py <url> [output_file.json]
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from scrapy.crawler import CrawlerProcess
from scrapy.settings import Settings
from scrapy import signals
from scrapy.signalmanager import dispatcher

# Add project root to path so src.helpers.scraper resolves
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.helpers.scraper import SiteSpider


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

        print(f"\n✓ Scraped {len(pages)} pages and saved to {output_path}")
        print(f"  File is ready for tokenization")
    else:
        print("\n⚠ No items were scraped")


if __name__ == "__main__":
    target_url = sys.argv[1] if len(sys.argv) > 1 else "https://www.torontomotors.ca/"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "scraped_data.json"
    run_spider(target_url, out_file)
