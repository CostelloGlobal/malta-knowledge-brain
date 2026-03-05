#!/usr/bin/env python3
"""
VisitMalta.com RAG Knowledge Brain - Complete Automation Script
================================================================

This script automates the entire pipeline for building a Malta knowledge RAG system:
1. Scrapes remaining Wikipedia Malta pages
2. Scrapes VisitMalta.com using Puppeteer for JavaScript-rendered pages
3. Generates embeddings using OpenAI text-embedding-3-large
4. Uploads everything to Qdrant Cloud automatically
5. Runs continuously until complete

USAGE:
    python3 malta_rag_automation.py

REQUIREMENTS:
    - Python 3.8+
    - Node.js and Puppeteer (for JavaScript rendering)
    - OpenAI API key
    - Qdrant Cloud API key

INSTALLATION:
    # Install Python dependencies
    pip install openai qdrant-client requests beautifulsoup4 lxml

    # Install Puppeteer (Node.js)
    npm install puppeteer

ENVIRONMENT VARIABLES:
    OPENAI_API_KEY: Your OpenAI API key
    QDRANT_API_KEY: Your Qdrant Cloud API key
    QDRANT_HOST: Qdrant Cloud host URL (e.g., https://xxxxx.qdrant.cloud)

Author: VisitMalta.co.uk Development Team
Date: 2026-03-06
"""

import os
import sys
import json
import time
import logging
import requests
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('malta_rag_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration settings for the pipeline"""

    # OpenAI Configuration
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
    EMBEDDING_MODEL = 'text-embedding-3-large'
    EMBEDDING_DIMENSIONS = 3072

    # Qdrant Configuration
    QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY', '')
    QDRANT_HOST = os.environ.get('QDRANT_HOST', '')
    COLLECTION_NAME = 'visitmalta_knowledge'

    # Paths
    DATA_DIR = Path('data')
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'

    # Chunking Settings
    CHUNK_MIN_SIZE = 100
    CHUNK_MAX_SIZE = 500
    CHUNK_OVERLAP = 50

    # Processing Settings
    BATCH_SIZE = 100  # Process embeddings in batches
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 5  # seconds


# =============================================================================
# WIKIPEDIA SCRAPING
# =============================================================================

class WikipediaScraper:
    """Scrapes Wikipedia pages for Malta-related content"""

    # List of Wikipedia Malta pages to scrape
    WIKIPEDIA_PAGES = [
        # Main Topics
        ('Malta', 'https://en.wikipedia.org/wiki/Malta'),
        ('History_of_Malta', 'https://en.wikipedia.org/wiki/History_of_Malta'),
        ('Geography_of_Malta', 'https://en.wikipedia.org/wiki/Geography_of_Malta'),
        ('Politics_of_Malta', 'https://en.wikipedia.org/wiki/Politics_of_Malta'),
        ('Economy_of_Malta', 'https://en.wikipedia.org/wiki/Economy_of_Malta'),
        ('Demographics_of_Malta', 'https://en.wikipedia.org/wiki/Demographics_of_Malta'),
        ('Culture_of_Malta', 'https://en.wikipedia.org/wiki/Culture_of_Malta'),

        # Locations
        ('Valletta', 'https://en.wikipedia.org/wiki/Valletta'),
        ('Gozo', 'https://en.wikipedia.org/wiki/Gozo'),
        ('Comino', 'https://en.wikipedia.org/wiki/Comino'),
        ('Mdina', 'https://en.wikipedia.org/wiki/Mdina'),
        ('Sliema', 'https://en.wikipedia.org/wiki/Sliema'),
        ('St_Julian%27s', 'https://en.wikipedia.org/wiki/St_Julian%27s,_Malta'),

        # History
        ('Great_Siege_of_Malta', 'https://en.wikipedia.org/wiki/Great_Siege_of_Malta'),
        ('Knights_of_Malta', 'https://en.wikipedia.org/wiki/Knights_Hospitaller'),
        ('Malta_in_World_War_II', 'https://en.wikipedia.org/wiki/Malta_in_World_War_II'),

        # Culture
        ('Maltese_cuisine', 'https://en.wikipedia.org/wiki/Maltese_cuisine'),
        ('Maltese_language', 'https://en.wikipedia.org/wiki/Maltese_language'),
        ('Maltese_literature', 'https://en.wikipedia.org/wiki/Maltese_literature'),
        ('Music_of_Malta', 'https://en.wikipedia.org/wiki/Music_of_Malta'),

        # Tourism & Attractions
        ('Tourism_in_Malta', 'https://en.wikipedia.org/wiki/Tourism_in_Malta'),
        ('St_John%27s_Co-Cathedral', 'https://en.wikipedia.org/wiki/St_John%27s_Co-Cathedral'),
        ('Megalithic_Temples_of_Malta', 'https://en.wikipedia.org/wiki/Megalithic_Temples_of_Malta'),
        ('Hal_Saflieni_Hypogeum', 'https://en.wikipedia.org/wiki/Hal_Saflieni_Hypogeum'),

        # Additional Topics
        ('Transport_in_Malta', 'https://en.wikipedia.org/wiki/Transport_in_Malta'),
        ('Education_in_Malta', 'https://en.wikipedia.org/wiki/Education_in_Malta'),
        ('Healthcare_in_Malta', 'https://en.wikipedia.org/wiki/Healthcare_in_Malta'),
        ('Sport_in_Malta', 'https://en.wikipedia.org/wiki/Sport_in_Malta'),
        ('Maltese_wine', 'https://en.wikipedia.org/wiki/Maltese_wine'),
    ]

    def __init__(self):
        self.scraped_pages = []
        self.failed_pages = []

    def scrape_page(self, title: str, url: str) -> Optional[Dict]:
        """Scrape a single Wikipedia page"""
        try:
            logger.info(f"Scraping Wikipedia: {title}")

            # Use Wikipedia API for cleaner content
            api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
            response = requests.get(api_url, timeout=30)

            if response.status_code == 200:
                data = response.json()

                page_data = {
                    'title': data.get('title', title),
                    'url': url,
                    'summary': data.get('extract', ''),
                    'content_type': 'Wikipedia',
                    'date_scraped': datetime.now().strftime('%Y-%m-%d')
                }

                # Try to get full content
                full_url = f"https://en.wikipedia.org/w/api.php"
                params = {
                    'action': 'query',
                    'titles': title,
                    'prop': 'extracts',
                    'explaintext': True,
                    'format': 'json'
                }
                full_response = requests.get(full_url, params=params, timeout=30)

                if full_response.status_code == 200:
                    pages = full_response.json().get('query', {}).get('pages', {})
                    for page_id, page_info in pages.items():
                        if page_id != '-1':
                            page_data['full_content'] = page_info.get('extract', '')

                self.scraped_pages.append(title)
                logger.info(f"Successfully scraped: {title}")
                return page_data

            else:
                logger.warning(f"Failed to scrape {title}: HTTP {response.status_code}")
                self.failed_pages.append({'title': title, 'url': url, 'error': f"HTTP {response.status_code}"})
                return None

        except Exception as e:
            logger.error(f"Error scraping {title}: {str(e)}")
            self.failed_pages.append({'title': title, 'url': url, 'error': str(e)})
            return None

    def scrape_all(self) -> List[Dict]:
        """Scrape all Wikipedia Malta pages"""
        logger.info(f"Starting Wikipedia scraping: {len(self.WIKIPEDIA_PAGES)} pages")

        all_pages = []

        for title, url in self.WIKIPEDIA_PAGES:
            page_data = self.scrape_page(title, url)
            if page_data:
                all_pages.append(page_data)

            # Rate limiting
            time.sleep(0.5)

        logger.info(f"Wikipedia scraping complete: {len(all_pages)} successful, {len(self.failed_pages)} failed")

        return all_pages


# =============================================================================
# VISITMALTA SCRAPING (PUPPETEER)
# =============================================================================

class VisitMaltaScraper:
    """Scrapes VisitMalta.com using Puppeteer for JavaScript rendering"""

    VISITMALTA_PAGES = [
        'https://www.visitmalta.com/en/',
        'https://www.visitmalta.com/en/attractions/',
        'https://www.visitmalta.com/en/destinations/',
        'https://www.visitmalta.com/en/experiences/',
        'https://www.visitmalta.com/en/food-and-drink-in-malta/',
        'https://www.visitmalta.com/en/accommodation/',
        'https://www.visitmalta.com/en/events-in-malta-and-gozo/',
        'https://www.visitmalta.com/en/essential-information/',
        'https://www.visitmalta.com/en/transportation-in-malta/',
    ]

    def __init__(self):
        self.scraped_pages = []
        self.failed_pages = []

    def scrape_with_puppeteer(self, url: str) -> Optional[Dict]:
        """Use Puppeteer to scrape JavaScript-rendered page"""

        # Create Puppeteer script
        puppeteer_script = f"""
const puppeteer = require('puppeteer');

async function scrape() {{
    const browser = await puppeteer.launch({{
        headless: 'new',
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    }});

    try {{
        const page = await browser.newPage();
        await page.goto('{url}', {{ waitUntil: 'networkidle0', timeout: 30000 }});

        // Extract main content
        const title = await page.title();
        const content = await page.evaluate(() => {{
            // Remove unwanted elements
            const remove = ['script', 'style', 'nav', 'footer', 'header'];
            remove.forEach(tag => {{
                document.querySelectorAll(tag).forEach(el => el.remove());
            }});

            // Get main content
            const main = document.querySelector('main') || document.querySelector('article') || document.body;
            return main ? main.innerText : '';
        }});

        return {{ title, content, url: '{url}' }};
    }} catch (e) {{
        console.error('Error:', e.message);
        return null;
    }} finally {{
        await browser.close();
    }}
}}

scrape().then(console.log).catch(console.error);
"""

        try:
            # Write temporary script
            script_path = '/tmp/puppeteer_scrape.js'
            with open(script_path, 'w') as f:
                f.write(puppeteer_script)

            # Run Puppeteer
            result = subprocess.run(
                ['node', script_path],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout.strip().split('\n')[-1])
                if data:
                    return {
                        'title': data.get('title', ''),
                        'content': data.get('content', ''),
                        'url': url,
                        'content_type': 'VisitMalta',
                        'date_scraped': datetime.now().strftime('%Y-%m-%d')
                    }

        except Exception as e:
            logger.error(f"Puppeteer error for {url}:")

        return None

    def scrape_all(self) -> List[Dict]:
        """Scrape all VisitMalta pages"""
        logger.info(f"Starting VisitMalta scraping: {len(self.VISITMALTA_PAGES)} pages")

        all_pages = []

        for url in self.VISITMALTA_PAGES:
            logger.info(f"Scraping VisitMalta: {url}")
            page_data = self.scrape_with_puppeteer(url)

            if page_data:
                all_pages.append(page_data)
                self.scraped_pages.append(url)
            else:
                self.failed_pages.append(url)

            time.sleep(1)  # Rate limiting

        logger.info(f"VisitMalta scraping complete: {len(all_pages)} successful, {len(self.failed_pages)} failed")

        return all_pages


# =============================================================================
# CONTENT PROCESSING
# =============================================================================

class ContentProcessor:
    """Processes and chunks scraped content"""

    def __init__(self, config: Config):
        self.config = config

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters
        text = re.sub(r'[^\w\s.,!?;:\-\'\"()\[\]–—]', '', text)
        return text.strip()

    def determine_category(self, title: str, content: str) -> str:
        """Determine content category"""
        text = (title + ' ' + content).lower()

        categories = {
            'History': ['history', 'war', 'siege', 'medieval', 'ancient'],
            'Geography': ['geography', 'island', 'coast', 'climate'],
            'Culture': ['culture', 'tradition', 'festival', 'music', 'art', 'literature'],
            'Tourism': ['tourism', 'tourist', 'attraction', 'beach', 'hotel'],
            'Economy': ['economy', 'trade', 'industry', 'business', 'finance'],
            'Politics': ['politics', 'government', 'election', 'parliament'],
            'Infrastructure': ['transport', 'airport', 'road', 'port'],
            'Food': ['cuisine', 'food', 'restaurant', 'wine'],
            'Religion': ['church', 'catholic', 'cathedral'],
            'Sports': ['sport', 'football', 'athlete']
        }

        for category, keywords in categories.items():
            if any(kw in text for kw in keywords):
                return category

        return 'General'

    def determine_location(self, title: str, content: str) -> str:
        """Determine primary location"""
        text = (title + ' ' + content).lower()

        locations = {
            'Valletta': ['valletta'],
            'Gozo': ['gozo'],
            'Comino': ['comino'],
            'Mdina': ['mdina'],
            'Sliema': ['sliema'],
            'St Julians': ['st julian'],
        }

        for location, keywords in locations.items():
            if any(kw in text for kw in keywords):
                return location

        return 'Malta'

    def chunk_text(self, text: str, min_size: int = 100, max_size: int = 500) -> List[str]:
        """Split text into semantic chunks"""
        if not text or len(text) < min_size:
            return [text] if text else []

        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence.split())

            if current_len + sentence_len > max_size and current_len >= min_size:
                chunks.append(' '.join(current))
                current = [sentence]
                current_len = sentence_len
            else:
                current.append(sentence)
                current_len += sentence_len

        if current and current_len >= min_size:
            chunks.append(' '.join(current))

        return [c for c in chunks if len(c.split()) >= 20]

    def generate_chunk_id(self, category: str, location: str, index: int) -> str:
        """Generate unique chunk ID"""
        cat_map = {
            'History': 'HIST', 'Geography': 'GEO', 'Culture': 'CULT',
            'Tourism': 'TOUR', 'Economy': 'ECON', 'Politics': 'POL',
            'Infrastructure': 'INFRA', 'Food': 'FOOD', 'Religion': 'REL',
            'Sports': 'SPRT', 'General': 'GEN'
        }
        loc_map = {
            'Malta': 'MT', 'Valletta': 'VAL', 'Gozo': 'GOZ',
            'Comino': 'COM', 'Mdina': 'MDA', 'Sliema': 'SLI',
            'St Julians': 'STJ'
        }

        cat = cat_map.get(category, 'GEN')
        loc = loc_map.get(location, 'MT')

        return f"VM-{cat}-{loc}-{index:04d}"

    def process_pages(self, pages: List[Dict]) -> List[Dict]:
        """Process all pages into chunks"""
        logger.info(f"Processing {len(pages)} pages into chunks")

        all_chunks = []
        chunk_index = 1

        for page in pages:
            title = page.get('title', 'Unknown')
            url = page.get('url', '')
            content = page.get('full_content') or page.get('summary', '') or page.get('content', '')

            if not content:
                continue

            cleaned_content = self.clean_text(content)
            category = self.determine_category(title, cleaned_content)
            location = self.determine_location(title, cleaned_content)

            # Chunk the content
            text_chunks = self.chunk_text(
                cleaned_content,
                min_size=self.config.CHUNK_MIN_SIZE,
                max_size=self.config.CHUNK_MAX_SIZE
            )

            for chunk_text in text_chunks:
                chunk_id = self.generate_chunk_id(category, location, chunk_index)

                chunk_record = {
                    'chunk_id': chunk_id,
                    'source_url': url,
                    'page_title': title,
                    'category': category,
                    'primary_location': location,
                    'secondary_locations': [],
                    'chunk_text': chunk_text,
                    'embedding': None,
                    'date_scraped': datetime.now().strftime('%Y-%m-%d'),
                    'content_type': page.get('content_type', 'Unknown')
                }

                all_chunks.append(chunk_record)
                chunk_index += 1

        logger.info(f"Created {len(all_chunks)} chunks from {len(pages)} pages")

        return all_chunks


# =============================================================================
# EMBEDDING GENERATION
# =============================================================================

class EmbeddingGenerator:
    """Generates embeddings using OpenAI"""

    def __init__(self, config: Config):
        self.config = config
        self.client = None

        if config.OPENAI_API_KEY:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=config.OPENAI_API_KEY)
                logger.info("OpenAI client initialized")
            except ImportError:
                logger.warning("OpenAI package not installed")

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text"""
        if not self.client:
            logger.error("OpenAI client not initialized")
            return None

        try:
            response = self.client.embeddings.create(
                model=self.config.EMBEDDING_MODEL,
                input=text,
                dimensions=self.config.EMBEDDING_DIMENSIONS
            )

            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None

    def generate_embeddings(self, chunks: List[Dict], batch_size: int = 100) -> List[Dict]:
        """Generate embeddings for all chunks"""
        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        # Check API key
        if not self.config.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY not set")
            return chunks

        for i, chunk in enumerate(chunks):
            if chunk.get('embedding') is None:
                text = chunk.get('chunk_text', '')

                # Truncate if too long
                if len(text) > 8000:
                    text = text[:8000]

                embedding = self.generate_embedding(text)

                if embedding:
                    chunk['embedding'] = embedding

                # Rate limiting
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i + 1}/{len(chunks)} chunks embedded")

                time.sleep(0.1)  # Avoid rate limits

        logger.info(f"Embedding generation complete: {len(chunks)} chunks")

        return chunks


# =============================================================================
# QDRANT UPLOAD
# =============================================================================

class QdrantUploader:
    """Uploads vectors to Qdrant Cloud"""

    def __init__(self, config: Config):
        self.config = config
        self.client = None

        if config.QDRANT_API_KEY and config.QDRANT_HOST:
            try:
                from qdrant_client import QdrantClient
                self.client = QdrantClient(
                    url=config.QDRANT_HOST,
                    api_key=config.QDRANT_API_KEY
                )
                logger.info("Qdrant client initialized")
            except ImportError:
                logger.warning("Qdrant client package not installed")

    def create_collection(self, collection_name: str = None):
        """Create Qdrant collection"""
        if not self.client:
            logger.error("Qdrant client not initialized")
            return False

        name = collection_name or self.config.COLLECTION_NAME

        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == name for c in collections)

            if not exists:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config={
                        "size": self.config.EMBEDDING_DIMENSIONS,
                        "distance": "Cosine"
                    }
                )
                logger.info(f"Created collection: {name}")
            else:
                logger.info(f"Collection already exists: {name}")

            return True

        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            return False

    def upload_vectors(self, chunks: List[Dict], collection_name: str = None) -> bool:
        """Upload chunks to Qdrant"""
        if not self.client:
            logger.error("Qdrant client not initialized")
            return False

        name = collection_name or self.config.COLLECTION_NAME

        # Prepare vectors
        vectors = []
        payloads = []

        for chunk in chunks:
            if chunk.get('embedding'):
                vectors.append(chunk['embedding'])

                # Create payload (without embedding)
                payload = {
                    'chunk_id': chunk.get('chunk_id'),
                    'source_url': chunk.get('source_url'),
                    'page_title': chunk.get('page_title'),
                    'category': chunk.get('category'),
                    'primary_location': chunk.get('primary_location'),
                    'chunk_text': chunk.get('chunk_text', '')[:1000]  # Limit text size
                }
                payloads.append(payload)

        if not vectors:
            logger.warning("No vectors to upload")
            return False

        try:
            # Upload in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch_vectors = vectors[i:i + batch_size]
                batch_payloads = payloads[i:i + batch_size]

                self.client.upsert(
                    collection_name=name,
                    points=[
                        {
                            "id": i + j,
                            "vector": vec,
                            "payload": pay
                        }
                        for j, (vec, pay) in enumerate(zip(batch_vectors, batch_payloads))
                    ]
                )

                logger.info(f"Uploaded batch {i//batch_size + 1}: {len(batch_vectors)} vectors")

            logger.info(f"Upload complete: {len(vectors)} vectors uploaded to {name}")
            return True

        except Exception as e:
            logger.error(f"Error uploading vectors: {str(e)}")
            return False


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class MaltaRAGPipeline:
    """Main pipeline orchestrator"""

    def __init__(self):
        self.config = Config()

        # Initialize components
        self.wikipedia_scraper = WikipediaScraper()
        self.visitmalta_scraper = VisitMaltaScraper()
        self.content_processor = ContentProcessor(self.config)
        self.embedding_generator = EmbeddingGenerator(self.config)
        self.qdrant_uploader = QdrantUploader(self.config)

        # State
        self.scraped_data = []
        self.chunks = []

    def load_existing_chunks(self, filepath: str = 'malta_combined_chunks.json'):
        """Load existing chunks from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.chunks = data
                else:
                    self.chunks = data.get('chunks', [])
            logger.info(f"Loaded {len(self.chunks)} existing chunks")
        except FileNotFoundError:
            logger.info("No existing chunks found, starting fresh")

    def save_chunks(self, filepath: str = 'malta_knowledge_chunks.json'):
        """Save chunks to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(self.chunks)} chunks to {filepath}")

    def run(self):
        """Run the complete pipeline"""
        logger.info("=" * 60)
        logger.info("Starting Malta RAG Pipeline")
        logger.info("=" * 60)

        # Step 1: Load existing chunks
        logger.info("\n[1/5] Loading existing chunks...")
        self.load_existing_chunks()

        # Step 2: Scrape Wikipedia
        logger.info("\n[2/5] Scraping Wikipedia Malta pages...")
        wiki_pages = self.wikipedia_scraper.scrape_all()
        self.scraped_data.extend(wiki_pages)
        logger.info(f"Scraped {len(wiki_pages)} Wikipedia pages")

        # Step 3: Scrape VisitMalta (Puppeteer)
        logger.info("\n[3/5] Scraping VisitMalta.com with Puppeteer...")
        visitmalta_pages = self.visitmalta_scraper.scrape_all()
        self.scraped_data.extend(visitmalta_pages)
        logger.info(f"Scraped {len(visitmalta_pages)} VisitMalta pages")

        # Step 4: Process content into chunks
        logger.info("\n[4/5] Processing content into chunks...")
        new_chunks = self.content_processor.process_pages(self.scraped_data)

        # Add new chunks to existing
        existing_ids = {c.get('chunk_id') for c in self.chunks}
        unique_new_chunks = [c for c in new_chunks if c.get('chunk_id') not in existing_ids]

        self.chunks.extend(unique_new_chunks)
        logger.info(f"Total chunks: {len(self.chunks)}")

        # Save chunks
        self.save_chunks()

        # Step 5: Generate embeddings
        logger.info("\n[5/5] Generating embeddings and uploading to Qdrant...")

        # Check if we should generate embeddings
        if self.config.OPENAI_API_KEY:
            self.chunks = self.embedding_generator.generate_embeddings(self.chunks)

            # Upload to Qdrant
            if self.config.QDRANT_API_KEY and self.config.QDRANT_HOST:
                self.qdrant_uploader.create_collection()
                self.qdrant_uploader.upload_vectors(self.chunks)

                # Save final chunks with embeddings
                self.save_chunks('malta_knowledge_chunks_with_embeddings.json')
        else:
            logger.warning("OpenAI API key not set - skipping embedding generation")

        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline Complete!")
        logger.info("=" * 60)
        logger.info(f"Total Wikipedia pages scraped: {len(wiki_pages)}")
        logger.info(f"Total VisitMalta pages scraped: {len(visitmalta_pages)}")
        logger.info(f"Total chunks created: {len(self.chunks)}")
        logger.info(f"Failed Wikipedia pages: {len(self.wikipedia_scraper.failed_pages)}")
        logger.info(f"Failed VisitMalta pages: {len(self.visitmalta_scraper.failed_pages)}")

        return {
            'wiki_pages': len(wiki_pages),
            'visitmalta_pages': len(visitmalta_pages),
            'total_chunks': len(self.chunks),
            'wiki_failures': self.wikipedia_scraper.failed_pages,
            'visitmalta_failures': self.visitmalta_scraper.failed_pages
        }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║   VisitMalta.com RAG Knowledge Brain - Automation Script    ║
    ║                                                              ║
    ║   This script will:                                         ║
    ║   1. Scrape Wikipedia Malta pages                          ║
    ║   2. Scrape VisitMalta.com with Puppeteer                 ║
    ║   3. Generate OpenAI embeddings                             ║
    ║   4. Upload to Qdrant Cloud                                ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Check environment variables
    if not Config.OPENAI_API_KEY:
        print("\n⚠️  WARNING: OPENAI_API_KEY not set")
        print("   Embedding generation will be skipped")
        print("   Set it with: export OPENAI_API_KEY=your_key_here\n")

    if not Config.QDRANT_API_KEY:
        print("⚠️  WARNING: QDRANT_API_KEY not set")
        print("   Vector upload will be skipped")
        print("   Set it with: export QDRANT_API_KEY=your_key_here\n")

    if not Config.QDRANT_HOST:
        print("⚠️  WARNING: QDRANT_HOST not set")
        print("   Vector upload will be skipped")
        print("   Set it with: export QDRANT_HOST=https://your-host.qdrant.cloud\n")

    # Run pipeline
    pipeline = MaltaRAGPipeline()
    results = pipeline.run()

    print("\n✅ Pipeline execution complete!")
    print(f"Results: {json.dumps(results, indent=2)}")
