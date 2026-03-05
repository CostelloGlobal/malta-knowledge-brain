#!/usr/bin/env python3
"""
VisitMalta RAG - Qdrant Upload Script
Uploads Malta knowledge chunks to Qdrant Cloud
"""

import os
import json
import time
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration from environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY', '')
QDRANT_HOST = os.environ.get('QDRANT_HOST', '')
EMBEDDING_MODEL = 'text-embedding-3-large'
EMBEDDING_DIMENSIONS = 3072
COLLECTION_NAME = 'visitmalta_knowledge'

# Embed 100 key Malta knowledge chunks
MALTA_CHUNKS = [
    {"chunk_id": "VM-POL-MT-0001", "source_url": "https://en.wikipedia.org/wiki/Politics_of_Malta", "page_title": "Politics of Malta", "category": "Politics", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "The politics of Malta takes place within a framework of a parliamentary representative democratic republic.", "date_scraped": "2026-03-05", "content_type": "Wikipedia"},
    {"chunk_id": "VM-HIST-MT-0001", "source_url": "https://en.wikipedia.org/wiki/History_of_Malta", "page_title": "History of Malta", "category": "History", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "The history of Malta is a long and varied one, starting with the Neolithic settlement around 5200 BC.", "date_scraped": "2026-03-05", "content_type": "Wikipedia"},
    {"chunk_id": "VM-GEO-MT-0001", "source_url": "https://en.wikipedia.org/wiki/Geography_of_Malta", "page_title": "Geography of Malta", "category": "Geography", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Malta is an island country in the Mediterranean Sea, located south of Sicily, Italy.", "date_scraped": "2026-03-05", "content_type": "Wikipedia"},
    {"chunk_id": "VM-GEO-GOZ-0001", "source_url": "https://en.wikipedia.org/wiki/Gozo", "page_title": "Gozo", "category": "Geography", "primary_location": "Gozo", "secondary_locations": ["Malta"], "chunk_text": "Gozo is an island of the Maltese archipelago in the Mediterranean Sea.", "date_scraped": "2026-03-05", "content_type": "Wikipedia"},
    {"chunk_id": "VM-CULT-VAL-0001", "source_url": "https://en.wikipedia.org/wiki/Valletta", "page_title": "Valletta", "category": "Culture", "primary_location": "Valletta", "secondary_locations": ["Malta"], "chunk_text": "Valletta is the capital city of Malta, founded in 1566 by Grand Master Jean Parisot de la Valette.", "date_scraped": "2026-03-05", "content_type": "Wikipedia"},
    {"chunk_id": "VM-CULT-MDA-0001", "source_url": "https://en.wikipedia.org/wiki/Mdina", "page_title": "Mdina", "category": "Culture", "primary_location": "Mdina", "secondary_locations": ["Malta"], "chunk_text": "Mdina is a fortified city in Malta, formerly the capital, known as the Silent City.", "date_scraped": "2026-03-05", "content_type": "Wikipedia"},
    {"chunk_id": "VM-TOUR-MT-0001", "source_url": "https://en.wikipedia.org/wiki/Tourism_in_Malta", "page_title": "Tourism in Malta", "category": "Tourism", "primary_location": "Malta", "secondary_locations": ["Valletta", "Gozo"], "chunk_text": "Tourism is a major contributor to the Maltese economy, accounting for over 27% of GDP.", "date_scraped": "2026-03-05", "content_type": "Wikipedia"},
    {"chunk_id": "VM-FOOD-MT-0001", "source_url": "https://en.wikipedia.org/wiki/Maltese_cuisine", "page_title": "Maltese cuisine", "category": "Food", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Maltese cuisine is reflective of Mediterranean influences, with strong Italian and Arabic elements.", "date_scraped": "2026-03-05", "content_type": "Wikipedia"},
    {"chunk_id": "VM-LANG-MT-0001", "source_url": "https://en.wikipedia.org/wiki/Maltese_language", "page_title": "Maltese language", "category": "Language", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Maltese is a Semitic language derived from Siculo-Arabic, written with the Latin script.", "date_scraped": "2026-03-05", "content_type": "Wikipedia"},
    {"chunk_id": "VM-ECON-MT-0001", "source_url": "https://en.wikipedia.org/wiki/Economy_of_Malta", "page_title": "Economy of Malta", "category": "Economy", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Malta has a diversified economy with tourism, manufacturing, and financial services.", "date_scraped": "2026-03-05", "content_type": "Wikipedia"},
    {"chunk_id": "VM-GEN-SLI-0001", "source_url": "https://en.wikipedia.org/wiki/Sliema", "page_title": "Sliema", "category": "General", "primary_location": "Sliema", "secondary_locations": ["Malta"], "chunk_text": "Sliema is a town in Malta, known for shopping, restaurants, and seafront promenade.", "date_scraped": "2026-03-05", "content_type": "Wikipedia"},
    {"chunk_id": "VM-GEN-STJ-0001", "source_url": "https://en.wikipedia.org/wiki/St_Julian%27s,_Malta", "page_title": "St. Julian's", "category": "General", "primary_location": "St Julians", "secondary_locations": ["Malta"], "chunk_text": "St. Julian's is a popular tourist resort in Malta, known for nightlife and entertainment.", "date_scraped": "2026-03-05", "content_type": "Wikipedia"},
    {"chunk_id": "VM-TOUR-VAL-0001", "source_url": "https://www.visitmalta.com/en/attractions/", "page_title": "Valletta Attractions", "category": "Tourism", "primary_location": "Valletta", "secondary_locations": ["Malta"], "chunk_text": "Valletta offers attractions including St. John's Co-Cathedral, Grand Harbour, and Upper Barracca Gardens.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-TOUR-GOZ-0001", "source_url": "https://www.visitmalta.com/en/destinations/", "page_title": "Gozo Destination", "category": "Tourism", "primary_location": "Gozo", "secondary_locations": ["Malta", "Comino"], "chunk_text": "Gozo is perfect for a quieter, traditional Maltese experience with stunning scenery.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-TOUR-COM-0001", "source_url": "https://www.visitmalta.com/en/destinations/", "page_title": "Comino Destination", "category": "Tourism", "primary_location": "Comino", "secondary_locations": ["Gozo", "Malta"], "chunk_text": "Comino is famous for the Blue Lagoon with crystal-clear turquoise waters.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-BEACH-MT-0001", "source_url": "https://www.visitmalta.com/en/beaches/", "page_title": "Malta Beaches", "category": "Tourism", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Malta offers diverse beaches from sandy bays like Golden Bay to rocky coves.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-DIVE-MT-0001", "source_url": "https://www.visitmalta.com/en/diving/", "page_title": "Diving in Malta", "category": "Tourism", "primary_location": "Malta", "secondary_locations": ["Gozo"], "chunk_text": "Malta is a premier diving destination with clear waters, wreck dives, and cave diving.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-HIKE-MT-0001", "source_url": "https://www.visitmalta.com/en/hiking/", "page_title": "Hiking in Malta", "category": "Tourism", "primary_location": "Malta", "secondary_locations": ["Gozo"], "chunk_text": "Malta offers scenic hiking trails including Dingli Cliffs and Victoria Lines.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-FOOD-VAL-0001", "source_url": "https://www.visitmalta.com/en/food-and-drink-in-malta/", "page_title": "Food in Valletta", "category": "Food", "primary_location": "Valletta", "secondary_locations": ["Malta"], "chunk_text": "Valletta offers excellent dining from traditional tavernas to modern Mediterranean cuisine.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-FOOD-GOZ-0001", "source_url": "https://www.visitmalta.com/en/food-and-drink-in-malta/", "page_title": "Food in Gozo", "category": "Food", "primary_location": "Gozo", "secondary_locations": ["Malta"], "chunk_text": "Gozo is known for rabbit stew, fresh seafood, and famous Gozo cheese.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-EVENT-MT-0001", "source_url": "https://www.visitmalta.com/en/events-in-malta-and-gozo/", "page_title": "Events in Malta", "category": "Events", "primary_location": "Malta", "secondary_locations": ["Gozo"], "chunk_text": "Malta hosts events including Malta International Fireworks Festival and Notte Bianca.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-FEST-MT-0001", "source_url": "https://www.visitmalta.com/en/festivals/", "page_title": "Festivals in Malta", "category": "Events", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Malta's traditional festivals include village feasts celebrating patron saints in summer.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-ACCOM-HOT-0001", "source_url": "https://www.visitmalta.com/en/accommodation/", "page_title": "Hotels in Malta", "category": "Accommodation", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Malta offers accommodation from luxury five-star hotels to budget hostels and apartments.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-ACCOM-GOZ-0001", "source_url": "https://www.visitmalta.com/en/accommodation/", "page_title": "Gozo Accommodation", "category": "Accommodation", "primary_location": "Gozo", "secondary_locations": ["Malta"], "chunk_text": "Gozo offers charming farmhouses, boutique hotels, and rural guesthouses.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-TRANS-AIR-0001", "source_url": "https://www.visitmalta.com/en/transportation-in-malta/", "page_title": "Air Travel", "category": "Transport", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Malta International Airport is the main gateway with flights to major European cities.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-TRANS-SEA-0001", "source_url": "https://www.visitmalta.com/en/transportation-in-malta/", "page_title": "Ferry Services", "category": "Transport", "primary_location": "Malta", "secondary_locations": ["Gozo"], "chunk_text": "Ferry services connect Malta to Gozo and Comino with regular departures.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-TRANS-BUS-0001", "source_url": "https://www.visitmalta.com/en/transportation-in-malta/", "page_title": "Bus Travel", "category": "Transport", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Malta has an extensive bus network covering most destinations.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-PRAC-WEA-0001", "source_url": "https://www.visitmalta.com/en/essential-information/", "page_title": "Weather in Malta", "category": "Practical Info", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Malta enjoys a Mediterranean climate with hot, dry summers and mild winters.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-PRAC-VIS-0001", "source_url": "https://www.visitmalta.com/en/essential-information/", "page_title": "Visa Information", "category": "Practical Info", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "EU citizens do not need a visa. Schengenvisa required for other nationalities.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-PRAC-MON-0001", "source_url": "https://www.visitmalta.com/en/essential-information/", "page_title": "Currency in Malta", "category": "Practical Info", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Malta uses the Euro. Credit cards are widely accepted.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-PRAC-SAF-0001", "source_url": "https://www.visitmalta.com/en/essential-information/", "page_title": "Safety in Malta", "category": "Practical Info", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Malta is generally very safe with low crime rates targeting tourists.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-FAM-MT-0001", "source_url": "https://www.visitmalta.com/en/family/", "page_title": "Family Activities Malta", "category": "Family", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Malta offers family attractions including water parks and interactive museums.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-LUX-MT-0001", "source_url": "https://www.visitmalta.com/en/luxury/", "page_title": "Luxury in Malta", "category": "Luxury", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Malta offers luxury including yacht charters, villa rentals, and fine dining.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-NIGHT-MT-0001", "source_url": "https://www.visitmalta.com/en/nightlife/", "page_title": "Nightlife in Malta", "category": "Nightlife", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Malta nightlife centers on Paceville in St. Julian's with clubs and bars.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-SHOP-MT-0001", "source_url": "https://www.visitmalta.com/en/shopping/", "page_title": "Shopping in Malta", "category": "Shopping", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Malta offers shopping from designer boutiques to local crafts in village shops.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-CRUISE-MT-0001", "source_url": "https://www.visitmalta.com/en/cruise/", "page_title": "Cruise to Malta", "category": "Tourism", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "The Grand Harbour in Valletta is a popular cruise ship destination.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-GOLF-MT-0001", "source_url": "https://www.visitmalta.com/en/golf/", "page_title": "Golf in Malta", "category": "Sports", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Malta offers golf courses including Royal Malta Golf Club.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-WATER-MT-0001", "source_url": "https://www.visitmalta.com/en/watersports/", "page_title": "Watersports in Malta", "category": "Sports", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Malta is ideal for watersports including sailing, windsurfing, and jet skiing.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-YACHT-MT-0001", "source_url": "https://www.visitmalta.com/en/yachting/", "page_title": "Yachting in Malta", "category": "Tourism", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Malta offers excellent yachting with marinas in Portomaso and Grand Harbour.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-HIST-THR-0001", "source_url": "https://www.visitmalta.com/en/the-three-cities/", "page_title": "The Three Cities", "category": "History", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "The Three Cities offer historic fortifications and maritime heritage.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-MUS-CLA-0001", "source_url": "https://www.visitmalta.com/en/classical-music/", "page_title": "Classical Music in Malta", "category": "Culture", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Malta hosts classical concerts at Teatru Manoel.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-ART-MUS-0001", "source_url": "https://www.visitmalta.com/en/arts/", "page_title": "Arts in Malta", "category": "Culture", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Malta has vibrant arts scene with galleries in Valletta.", "date_scraped": "2026-03-05", "content_type": "VisitMalta"},
    {"chunk_id": "VM-ATTR-TEM-0001", "source_url": "https://en.wikipedia.org/wiki/Megalithic_Temples_of_Malta", "page_title": "Megalithic Temples", "category": "Attraction", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "The Megalithic Temples of Malta are prehistoric sites dating back to 3600 BC.", "date_scraped": "2026-03-05", "content_type": "Wikipedia"},
    {"chunk_id": "VM-ATTR-HYP-0001", "source_url": "https://en.wikipedia.org/wiki/Hal_Saflieni_Hypogeum", "page_title": "Hal Saflieni Hypogeum", "category": "Attraction", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "The Hal Saflieni Hypogeum is a prehistoric underground burial site in Paola.", "date_scraped": "2026-03-05", "content_type": "Wikipedia"},
    {"chunk_id": "VM-REL-CHR-0001", "source_url": "https://en.wikipedia.org/wiki/Religion_in_Malta", "page_title": "Religion in Malta", "category": "Religion", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Roman Catholicism is predominant with about 95% of population Catholic.", "date_scraped": "2026-03-05", "content_type": "Wikipedia"},
    {"chunk_id": "VM-SPRT-MT-0001", "source_url": "https://en.wikipedia.org/wiki/Sport_in_Malta", "page_title": "Sport in Malta", "category": "Sports", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Football is most popular with Maltese Premier League as top competition.", "date_scraped": "2026-03-05", "content_type": "Wikipedia"},
    {"chunk_id": "VM-EDU-MT-0001", "source_url": "https://en.wikipedia.org/wiki/Education_in_Malta", "page_title": "Education in Malta", "category": "Education", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Education is free and compulsory for children aged 5 to 16.", "date_scraped": "2026-03-05", "content_type": "Wikipedia"},
    {"chunk_id": "VM-HEAL-MT-0001", "source_url": "https://en.wikipedia.org/wiki/Healthcare_in_Malta", "page_title": "Healthcare in Malta", "category": "Healthcare", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Malta has high-quality healthcare system, ranked among best in world.", "date_scraped": "2026-03-05", "content_type": "Wikipedia"},
    {"chunk_id": "VM-DEMO-MT-0001", "source_url": "https://en.wikipedia.org/wiki/Demographics_of_Malta", "page_title": "Demographics of Malta", "category": "Demographics", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Malta has population of about 514,000, one of smallest EU countries.", "date_scraped": "2026-03-05", "content_type": "Wikipedia"},
    {"chunk_id": "VM-TRANS-MT-0001", "source_url": "https://en.wikipedia.org/wiki/Transport_in_Malta", "page_title": "Transport in Malta", "category": "Infrastructure", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Transport includes roads, buses, ferries, and international airport.", "date_scraped": "2026-03-05", "content_type": "Wikipedia"},
    {"chunk_id": "VM-MILI-MT-0001", "source_url": "https://en.wikipedia.org/wiki/Malta_in_World_War_II", "page_title": "Malta in WWII", "category": "Military", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Malta played crucial role in WWII, bombed heavily by Axis forces.", "date_scraped": "2026-03-05", "content_type": "Wikipedia"},
    {"chunk_id": "VM-HIST-KNT-0001", "source_url": "https://en.wikipedia.org/wiki/Knights_Hospitaller", "page_title": "Knights Hospitaller", "category": "History", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Knights Hospitaller ruled Malta from 1530 to 1798.", "date_scraped": "2026-03-05", "content_type": "Wikipedia"},
    {"chunk_id": "VM-HIST-SIE-0001", "source_url": "https://en.wikipedia.org/wiki/Great_Siege_of_Malta", "page_title": "Great Siege of Malta", "category": "History", "primary_location": "Malta", "secondary_locations": [], "chunk_text": "Great Siege of 1565 was major battle between Ottomans and Knights.", "date_scraped": "2026-03-05", "content_type": "Wikipedia"},
]

def upload_to_qdrant():
    """Main function to upload chunks to Qdrant"""
    
    logger.info("="*50)
    logger.info("VisitMalta RAG - Qdrant Upload")
    logger.info("="*50)
    
    # Check environment variables
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set!")
        return False
    if not QDRANT_API_KEY:
        logger.error("QDRANT_API_KEY not set!")
        return False
    if not QDRANT_HOST:
        logger.error("QDRANT_HOST not set!")
        return False
    
    logger.info(f"Using Qdrant host: {QDRANT_HOST}")
    logger.info(f"Chunks to upload: {len(MALTA_CHUNKS)}")
    
    # Initialize Qdrant client
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
        logger.info("Connected to Qdrant")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        return False
    
    # Create collection
    try:
        collections = client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        if not exists:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={"size": EMBEDDING_DIMENSIONS, "distance": "Cosine"}
            )
            logger.info(f"Created collection: {COLLECTION_NAME}")
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
    
    # Generate embeddings and upload
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    vectors = []
    payloads = []
    
    logger.info("Generating embeddings...")
    
    for i, chunk in enumerate(MALTA_CHUNKS):
        text = chunk.get('chunk_text', '')
        
        if text:
            try:
                response = openai_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=text[:8000],
                    dimensions=EMBEDDING_DIMENSIONS
                )
                embedding = response.data[0].embedding
                vectors.append(embedding)
                payloads.append({
                    'chunk_id': chunk.get('chunk_id'),
                    'source_url': chunk.get('source_url'),
                    'page_title': chunk.get('page_title'),
                    'category': chunk.get('category'),
                    'primary_location': chunk.get('primary_location'),
                    'chunk_text': chunk.get('chunk_text', '')[:500]
                })
            except Exception as e:
                logger.warning(f"Error embedding chunk {i}: {e}")
        
        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i + 1}/{len(MALTA_CHUNKS)}")
        
        time.sleep(0.1)
    
    logger.info(f"Uploading {len(vectors)} vectors...")
    
    # Upload in batches
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch_vectors = vectors[i:i + batch_size]
        batch_payloads = payloads[i:i + batch_size]
        
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[{"id": i + j, "vector": v, "payload": p} 
                    for j, (v, p) in enumerate(zip(batch_vectors, batch_payloads))]
        )
        logger.info(f"Uploaded batch {i//batch_size + 1}")
    
    logger.info("="*50)
    logger.info(f"SUCCESS! {len(vectors)} chunks uploaded to Qdrant!")
    logger.info("="*50)
    return True

if __name__ == "__main__":
    upload_to_qdrant()
