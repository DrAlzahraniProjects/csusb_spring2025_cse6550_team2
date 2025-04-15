import scrapy
import re
import os
import hashlib
import pathlib
from urllib.parse import urlparse

# Use asyncio-compatible Twisted reactor
os.environ["TWISTED_REACTOR"] = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
import twisted.internet.asyncioreactor
twisted.internet.asyncioreactor.install()

# Ensure necessary directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("data/cache", exist_ok=True)

# Precompile regex patterns for performance
TAG_RE = re.compile(r'<[^>]+>')  # Matches HTML tags
WHITESPACE_RE = re.compile(r'\s+')  # Matches all types of whitespace

def clean_text(text):
    """Remove HTML tags and normalize whitespace."""
    text = TAG_RE.sub('', text)
    return WHITESPACE_RE.sub(' ', text).strip()

def segment_text(text, max_chunk_size=512):
    """Split text into smaller chunks for indexing or embedding."""
    return [text[i:i + max_chunk_size].strip() for i in range(0, len(text), max_chunk_size)]

def hash_text(text: str) -> str:
    """Compute MD5 hash of the given text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def get_cache_path(url: str) -> pathlib.Path:
    """Generate a safe cache filename based on the URL hash."""
    safe_name = hashlib.md5(url.encode("utf-8")).hexdigest()
    return pathlib.Path("data/cache") / f"{safe_name}.txt"

class GoAbroadSpider(scrapy.Spider):
    name = "goabroad"
    allowed_domains = ["goabroad.csusb.edu"]
    start_urls = ["https://goabroad.csusb.edu/"]

    # Configure polite crawling behavior
    custom_settings = {
        "DOWNLOAD_DELAY": 1,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 1,
        "AUTOTHROTTLE_MAX_DELAY": 3,
    }

    def parse(self, response):
        self.logger.info(f"Parsing URL: {response.url}")
        url = response.url
        title = response.xpath("//title/text()").get(default="").strip()
        meta_description = response.xpath("//meta[@name='description']/@content").get(default="").strip()
        structured_data = response.xpath("//script[@type='application/ld+json']/text()").getall()

        # Extract and join all visible text from the page body
        raw_text_nodes = response.xpath("//body//text()[normalize-space()]").getall()
        joined_text = " ".join(text.strip() for text in raw_text_nodes if text.strip())

        # Clean and hash the page content for change detection
        cleaned_text = clean_text(joined_text)
        content_hash = hash_text(cleaned_text)
        cache_path = get_cache_path(response.url)

        # Skip this page if content hasn't changed since last crawl
        if cache_path.exists():
            with open(cache_path, "r") as f:
                if f.read().strip() == content_hash:
                    self.logger.info(f"[SKIPPED] No change for {url}")
                    return

        # Content is new or updated — save hash to cache
        with open(cache_path, "w") as f:
            f.write(content_hash)

        # Split cleaned text into segments
        segments = segment_text(cleaned_text)

        # Discover all internal links for recursive crawling
        internal_links = response.css("a::attr(href)").getall()
        internal_links = [
            response.urljoin(link) for link in internal_links
            if urlparse(response.urljoin(link)).hostname and (
                urlparse(response.urljoin(link)).hostname == "goabroad.csusb.edu"
                or urlparse(response.urljoin(link)).hostname.endswith(".goabroad.csusb.edu")
            )
        ]
        internal_links = list(set(internal_links))  # Remove duplicates

        # Output parsed data
        yield {
            "url": url,
            "title": title,
            "meta_description": meta_description,
            "structured_data": structured_data,
            "cleaned_text": cleaned_text,
            "segments": segments,
            "internal_links": internal_links,
        }

        # Recursively follow internal links
        for link in internal_links:
            yield scrapy.Request(url=link, callback=self.parse)
