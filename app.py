from apscheduler.schedulers.background import BackgroundScheduler
from hashlib import md5
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from urllib.parse import urlparse
import json
import os
import re
import scrapy
import scrapy.crawler
import scrapy.http.response
import subprocess

# Constants
SITE = "goabroad.csusb.edu"
SEGMENT_SIZE: int = 512  # Customize as needed
URL_HASHES_PATH: str = os.path.join(".", "data", "index", "hashes.json")
VECTORSTORE_DIR: str = os.path.join(".", "data", "index")

# Compile regex for HTML cleaning
TAG_RE = re.compile(r'<[^>]+>')
WHITESPACE_RE = re.compile(r'\s+')

# Ensure hash storage exists
URL_HASHES: dict[str, str] = {}
if os.path.exists(URL_HASHES_PATH):
    with open(URL_HASHES_PATH, "r") as f:
        try:
            URL_HASHES: dict[str, str] = json.load(f)
        except:
            pass

# MD5 hash function
def generate_md5_hash(text: str) -> str:
    return md5(text.encode("utf-8")).hexdigest()

# Initialize embeddings & FAISS vectorstore
embedding = FastEmbedEmbeddings()
if os.path.exists(VECTORSTORE_DIR):
    vectorstore = FAISS.load_local(VECTORSTORE_DIR, embedding, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_texts([], embedding=embedding)
    vectorstore.save_local(VECTORSTORE_DIR)

# Scrapy spider
class GoAbroadSpider(scrapy.Spider):
    name = "goabroad"
    allowed_domains = [SITE]
    start_urls = [f"https://{SITE}/"]
    custom_settings = {
        "DOWNLOAD_DELAY": 0.5,
        "DEPTH_LIMIT": 3,
        "LOG_LEVEL": "INFO",
    }

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.page_count = 0
    #     self.start_time = time.time()
    #     self.timeout = 600

    def parse(self, response: scrapy.http.response.Response):
        # if time.time() - self.start_time > self.timeout:
        #     self.logger.info("Timeout reached. Stopping.")
        #     raise scrapy.exceptions.CloseSpider("timeout")

        # self.page_count += 1
        # self.logger.info(f"Crawling page {self.page_count}: {response.url}")

        global URL_HASHES

        # Extract and clean text
        raw_text_nodes = response.xpath("//body//text()[normalize-space()]").getall()
        joined_text = " ".join(text.strip() for text in raw_text_nodes if text.strip())
        cleaned_text = WHITESPACE_RE.sub(' ', TAG_RE.sub('', joined_text)).strip()

        content_hash = generate_md5_hash(cleaned_text)
        if URL_HASHES.get(response.url) == content_hash:
            self.logger.info(f"[SKIPPED] No change for {response.url}")
        else:
            URL_HASHES[response.url] = content_hash
            os.makedirs(os.path.dirname(URL_HASHES_PATH), exist_ok=True)
            with open(URL_HASHES_PATH, "w") as f:
                json.dump(URL_HASHES, f)

            # Split into segments and store
            segments = {cleaned_text[i:i + SEGMENT_SIZE].strip() for i in range(0, len(cleaned_text), SEGMENT_SIZE)}
            self.logger.info(f"Adding {len(segments)} segments for {response.url}")
            vectorstore.add_texts(list(segments), metadatas=[{"url": response.url} for _ in segments])
            vectorstore.save_local(VECTORSTORE_DIR)

        # Recursively follow internal links
        internal_links = response.css("a::attr(href)").getall()
        internal_links = list({
            response.urljoin(link)
            for link in internal_links
            if urlparse(response.urljoin(link)).hostname and
               (urlparse(response.urljoin(link)).hostname == SITE
                or urlparse(response.urljoin(link)).hostname.endswith("." + SITE))
        })
        self.logger.info(f"Found {len(internal_links)} internal links.")
        for link in internal_links:
            yield scrapy.Request(url=link, callback=self.parse)

# Run crawler once
def runScraper() -> None:
    process = scrapy.crawler.CrawlerProcess({
        'USER_AGENT': 'Mozilla/5.0 (compatible; GoAbroadSpider/1.0)',
        'LOG_LEVEL': 'INFO',
    })
    process.crawl(GoAbroadSpider)
    process.start()


scheduler = BackgroundScheduler()
scheduler.add_job(runScraper, "interval", hours=24)
scheduler.start()
subprocess.Popen(["apache2ctl", "start"])
subprocess.run(["streamlit", "run", "chatbot.py", "--server.baseUrlPath=/team2s25", "--server.port=2502", "--theme.backgroundColor=#0065BD", "--theme.primaryColor=#808284", "--theme.secondaryBackgroundColor=#808284", "--theme.textColor=#FFFFFF", "--browser.gatherUsageStats=false"])