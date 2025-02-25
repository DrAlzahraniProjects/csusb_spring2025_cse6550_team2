import scrapy
import re
import os

# Set the Twisted reactor to use asyncio.
os.environ["TWISTED_REACTOR"] = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
import twisted.internet.asyncioreactor
twisted.internet.asyncioreactor.install()

# Ensure the data directory exists.
os.makedirs("data", exist_ok=True)

# Precompile regex patterns for efficiency.
TAG_RE = re.compile(r'<[^>]+>')
WHITESPACE_RE = re.compile(r'\s+')

def clean_text(text):
    """
    Remove residual HTML tags and collapse extra whitespace.
    """
    text = TAG_RE.sub('', text)
    return WHITESPACE_RE.sub(' ', text).strip()

def segment_text(text, max_chunk_size=512):
    """
    Split the cleaned text into smaller segments.
    """
    return [text[i:i + max_chunk_size].strip() for i in range(0, len(text), max_chunk_size)]

class GoAbroadSpider(scrapy.Spider):
    name = "goabroad"
    allowed_domains = ["goabroad.csusb.edu"]
    start_urls = ["https://goabroad.csusb.edu/"]

    # Custom settings for politeness.
    custom_settings = {
        "DOWNLOAD_DELAY": 1,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 1,
        "AUTOTHROTTLE_MAX_DELAY": 3,
    }

    def parse(self, response):
        self.logger.info(f"Parsing URL: {response.url}")
        # Gather reference information.
        url = response.url
        title = response.xpath("//title/text()").get(default="").strip()
        meta_description = response.xpath("//meta[@name='description']/@content").get(default="").strip()

        # Extract structured data (e.g., JSON‑LD).
        structured_data = response.xpath("//script[@type='application/ld+json']/text()").getall()

        # Extract text nodes from the body of the page.
        raw_text_nodes = response.xpath("//body//text()[normalize-space()]").getall()
        joined_text = " ".join(text.strip() for text in raw_text_nodes if text.strip())

        # Clean the text.
        cleaned_text = clean_text(joined_text)

        # Segment the cleaned text into chunks.
        segments = segment_text(cleaned_text, max_chunk_size=512)

        # Extract and normalize internal links for further crawling.
        internal_links = response.css("a::attr(href)").getall()
        internal_links = [response.urljoin(link) for link in internal_links if "goabroad.csusb.edu" in response.urljoin(link)]
        # Deduplicate internal links.
        internal_links = list(set(internal_links))

        yield {
            "url": url,
            "title": title,
            "meta_description": meta_description,
            "structured_data": structured_data,
            "cleaned_text": cleaned_text,
            "segments": segments,
            "internal_links": internal_links,
        }

        # Follow internal links to continue crawling the site.
        for link in internal_links:
            yield scrapy.Request(url=link, callback=self.parse)
