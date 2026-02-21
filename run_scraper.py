"""Root convenience script — delegates to src/helpers/run_scraper.py"""
import sys
from src.helpers.run_scraper import run_spider

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.torontomotors.ca/"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "scraped_data.json"
    run_spider(url, output_file)
