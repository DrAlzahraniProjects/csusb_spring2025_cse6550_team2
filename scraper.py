import scrapy
from scrapy.crawler import CrawlerProcess
import os
import re
import hashlib
import json
from urllib.parse import urlparse
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document # Import Document type
from faiss import IndexFlatL2 # Import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore # Import InMemoryDocstore


# Constants (Duplicate relevant ones or read from config if needed)
# For this example, we'll duplicate the needed constants
URL_HASHES_PATH = os.path.join(".", "data", "index", "hashes.json")
SEGMENT_SIZE = 512
INDEX_PATH = os.path.join(".", "data", "index")

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs(INDEX_PATH, exist_ok=True)


# Global variable for hashes - loaded at script startup
URL_HASHES: dict[str, str] = {}
if os.path.exists(URL_HASHES_PATH):
    with open(URL_HASHES_PATH, "r") as f:
        try:
            URL_HASHES = json.load(f)
        except json.JSONDecodeError:
            URL_HASHES = {}
            print(f"Warning: Could not decode JSON from {URL_HASHES_PATH}. Starting with empty hashes.")
        except Exception as e:
             URL_HASHES = {}
             print(f"Error loading URL hashes from {URL_HASHES_PATH}: {e}. Starting with empty hashes.")


# Initialize embedding model
EMBEDDING_MODEL = FastEmbedEmbeddings()

# Regex for cleaning text
TAG_RE = re.compile(r'<[^>]+>')
WHITESPACE_RE = re.compile(r'\s+')

def generate_md5_hash(text: str) -> str: # Changed param name from question to text for clarity
    """Generate MD5 hash for any input text"""
    if not isinstance(text, str) or not text:
         # Return a predictable hash for empty/invalid input
         return hashlib.md5(b"").hexdigest()
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# Use typing hints for clarity
def load_vectorstore(index_path: str, embeddings: FastEmbedEmbeddings) -> FAISS | None:
    """Loads the FAISS vectorstore from disk."""
    if not os.path.exists(index_path):
        print(f"Info: Index path does not exist: {index_path}. Will create a new index.")
        return None # Indicate that no existing index was found
    try:
        # Check if the required FAISS files exist before attempting to load
        index_file = os.path.join(index_path, "index.faiss")
        docstore_file = os.path.join(index_path, "docstore.json")
        if not os.path.exists(index_file) or not os.path.exists(docstore_file):
            print(f"Info: FAISS index files not found in {index_path}. Will create a new index.")
            return None # Indicate that necessary files are missing

        print(f"Attempting to load existing vectorstore from {index_path}...")
        # allow_dangerous_deserialization is needed because FAISS pickles Python objects
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        print("Successfully loaded existing vectorstore.")
        return vectorstore
    except Exception as e:
        print(f"Error loading vectorstore from {index_path}: {e}")
        print("Will create a new index.")
        return None # Indicate failure to load

# Use typing hints for clarity
def create_or_update_vectorstore(documents: list[Document], embeddings: FastEmbedEmbeddings, index_path: str) -> FAISS:
    """Creates a new or updates an existing FAISS vectorstore."""
    vectorstore = load_vectorstore(index_path, embeddings)

    if vectorstore is None:
        # Create a new vectorstore if loading failed or no index exists
        if documents:
            print(f"Creating new vectorstore with {len(documents)} documents.")
            # Use from_documents to create the index from scratch
            vectorstore = FAISS.from_documents(documents, embeddings)
        else:
            print("No documents to index. Creating an empty vectorstore structure.")
            # Create an empty vectorstore with a dummy embedding and docstore
            # Get the dimension from the embedding model
            # Need to embed something to get the dimension if the model is lazy
            try:
                 dummy_embedding = embeddings.embed_query("dummy")
                 dimension = len(dummy_embedding)
            except Exception as e:
                 print(f"Error getting embedding dimension: {e}. Using a default dimension (768) - adjust if needed.")
                 # Default dimension for FastEmbed models like BGE-small-en-v1.5
                 dimension = 768 # Or raise an error if dimension cannot be determined

            # Initialize an empty FAISS index
            index = IndexFlatL2(dimension)
            # Initialize an empty docstore
            docstore = InMemoryDocstore({})
            # Create an empty FAISS object
            # The FAISS constructor signature requires the embedding function, the index, the docstore, and index_to_docstore_id
            vectorstore = FAISS(embeddings.embed_query, index, docstore, {})


    else:
        # Update the existing vectorstore
        print(f"Checking for existing documents in the vectorstore...")
        # Langchain's FAISS.add_documents method can handle adding new documents.
        # It internally adds documents to the docstore and updates the FAISS index.
        if documents:
            print(f"Adding {len(documents)} new/updated documents to the vectorstore.")
            vectorstore.add_documents(documents)
            print(f"Added {len(documents)} documents to the vectorstore.")
        else:
            print("No new or updated documents to add to the vectorstore.")


    # Save the (new or updated) vectorstore
    try:
        print(f"Saving vectorstore to {index_path}...")
        vectorstore.save_local(index_path)
        print("Vectorstore saved.")
    except Exception as e:
         print(f"Error saving vectorstore to {index_path}: {e}")
         # Decide how to handle save errors - might want to exit or retry


    return vectorstore

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
        # Disable default scrapy pipelines if they interfere
        # 'ITEM_PIPELINES': {},
        # Store scraped items in a list within the spider instance
        'ITEM_PIPELINES': {'__main__.DocumentCollectorPipeline': 100},
    }

    # Initialize a list to hold collected documents
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scraped_documents = []

    def parse(self, response):
        global URL_HASHES
        global URL_HASHES_PATH

        # Use response.css or response.xpath to select relevant content instead of the entire body
        # This makes the scraper more robust to irrelevant text on the page.
        # Example: selecting text within main content areas if identifiable by CSS classes or IDs
        # For now, keeping the broad body selection as in the original code, but be aware this can pick up noise.
        raw_text_nodes = response.xpath("//body//text()[normalize-space()]").getall()
        joined_text = " ".join(text.strip() for text in raw_text_nodes if text.strip())
        cleaned_text = WHITESPACE_RE.sub(' ', TAG_RE.sub('', joined_text)).strip()

        content_hash = generate_md5_hash(cleaned_text)

        # Check if content has changed or if URL is new
        if URL_HASHES.get(response.url) != content_hash:
            print(f"Content changed or new URL: {response.url}. Updating hash and collecting segments.")
            # Update in-memory cache
            URL_HASHES[response.url] = content_hash

            # Persist the updated hash table to disk immediately (or batch later)
            # For simplicity, we save on each change. Batching might be more efficient.
            try:
                with open(URL_HASHES_PATH, "w") as f:
                    json.dump(URL_HASHES, f, indent=2)
            except Exception as e:
                print(f"Error saving URL hashes to {URL_HASHES_PATH}: {e}")

            # Segment the cleaned text into chunks.
            segments = []
            for i in range(0, len(cleaned_text), SEGMENT_SIZE):
                 segment_text = cleaned_text[i:i + SEGMENT_SIZE].strip()
                 if segment_text: # Only add non-empty segments
                     segments.append(Document(page_content=segment_text, metadata={"url": response.url}))

            # Instead of adding directly to vectorstore here, yield the documents
            # A pipeline will collect these documents
            for doc in segments:
                 yield doc


        # Extract and normalize internal links for further crawling.
        internal_links = response.css("a::attr(href)").getall()
        # Filter unique internal links belonging to the allowed domain
        internal_links = list(
            {response.urljoin(link)
             for link in internal_links
             if urlparse(response.urljoin(link)).hostname is not None and
                (urlparse(response.urljoin(link)).hostname == "goabroad.csusb.edu" or
                 urlparse(response.urljoin(link)).hostname.endswith(".goabroad.csusb.edu"))
            }
        )

        # Yield new Requests to follow links
        for link in internal_links:
            yield scrapy.Request(url=link, callback=self.parse)

    # Removed the closed method that directly updated the vectorstore

# Define a simple pipeline to collect documents
class DocumentCollectorPipeline:
    def process_item(self, item, spider):
        # Assuming the spider yields Document objects directly
        if isinstance(item, Document):
            spider.scraped_documents.append(item)
        return item # Return the item to potentially be processed by other pipelines (none defined here)


if __name__ == "__main__":
    print("Starting scraper...")
    # Use CrawlerProcess to run the spider
    # Pass the spider class directly
    process = CrawlerProcess({
        'USER_AGENT': 'CSUSB_Education_Abroad_Chatbot_Scraper (+https://goabroad.csusb.edu/)', # More descriptive User-Agent
        'LOG_LEVEL': 'INFO', # Reduce log verbosity
        'ROBOTSTXT_OBEY': True, # Respect robots.txt
        'CONCURRENT_REQUESTS': 4, # Limit concurrent requests
        'AUTOTHROTTLE_ENABLED': True, # Ensure politeness
        'AUTOTHROTTLE_START_DELAY': 1,
        'AUTOTHROTTLE_MAX_DELAY': 3,
        'DOWNLOAD_DELAY': 1, # Politeness delay
        'ITEM_PIPELINES': {'__main__.DocumentCollectorPipeline': 100}, # Use the pipeline to collect items
        'EXTENSIONS': {'scrapy.extensions.logstats.LogStats': 500}, # Keep some useful extensions
        # 'STATS_DUMP': False, # Disable dumping stats at the end
    })

    # Schedule the spider to run
    # The CrawlerProcess will collect items yielded by the spider into the spider instance
    process.crawl(GoAbroadSpider)

    # Start the crawling process
    # This is a blocking call. It runs the spider and all its scheduled tasks.
    process.start()

    print("Scraper finished.")

    # After the process finishes, the spider instance is available in the process.crawler.spider list
    # We can access the collected documents here
    # Get the crawler for our spider instance
    crawler = list(process.crawlers)[0] # Assuming only one spider was crawled
    spider_instance = crawler.spider

    if hasattr(spider_instance, 'scraped_documents') and spider_instance.scraped_documents:
        print(f"Total documents collected by pipeline: {len(spider_instance.scraped_documents)}")
        # Create or update the vectorstore after the crawl
        create_or_update_vectorstore(spider_instance.scraped_documents, EMBEDDING_MODEL, INDEX_PATH)
    else:
        print("No documents were collected during the crawl.")