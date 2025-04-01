import requests
from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup
from langdetect import detect
import csv
from tqdm import tqdm
import re
import os
import gzip
import json
from urllib.parse import urlparse
from multiprocessing import Process, Queue, Manager, Lock
from datetime import datetime
import random  # Add this to the imports at the top
import unicodedata
import pandas as pd
from io import StringIO
import gc  # Add to imports

# Helper function to extract domain from URL
def extract_domain(url):
    return urlparse(url).netloc

# Helper function to extract a snippet from HTML
def extract_snippet(soup, max_sentences=2):
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
        tag.decompose()
    text = soup.get_text(separator=' ', strip=True)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return ' '.join(sentences[:max_sentences])

# Update the worker function with better content detection
def worker(input_queue, output_queue, domain_counts, lock, max_per_domain):
    while True:
        item = input_queue.get()
        if item is None:  # Stop signal
            break
        try:
            # Unpack the pre-extracted data
            url, html, warc_record_id, content_type, server, warc_date = item
            
            domain = extract_domain(url)
            with lock:
                if domain in domain_counts and domain_counts[domain] >= max_per_domain:
                    continue
            
            # Try to determine if content is XML
            is_xml = html.strip().startswith('<?xml') or html.strip().startswith('<rss')
            parser = 'xml' if is_xml else 'lxml'
            features = {'features': parser} if is_xml else {}
            
            # Only create soup object when needed
            if html and url:
                # Use faster 'html.parser' instead of 'lxml' when possible
                soup = BeautifulSoup(html, 'html.parser' if not is_xml else 'lxml', **features)
                title = soup.title.get_text(strip=True) if soup.title else ''
                snippet = extract_snippet(soup)
                
                # Extract meta description
                meta_desc = ""
                meta_tag = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
                if meta_tag and meta_tag.get('content'):
                    meta_desc = meta_tag['content']
                
                if not title or not snippet:
                    continue
                
                # Only try language detection on reasonable-length text
                if len(snippet) > 10:
                    try:
                        language = detect(snippet)
                        output_queue.put((url, title, snippet, language, warc_record_id, content_type, server, warc_date, meta_desc))
                        with lock:
                            domain_counts[domain] = domain_counts.get(domain, 0) + 1
                    except:
                        continue
        except Exception as e:
            continue

# Writer function to save results to Parquet
def writer(output_queue, output_file, append=False, batch_size=1000):
    # Replace .csv with .parquet in filename if needed
    if output_file.endswith('.csv'):
        output_file = output_file[:-4] + '.parquet'
    
    # Create lists to accumulate data
    urls = []
    titles = []
    snippets = []
    languages = []
    warc_ids = []
    warc_dates = []
    meta_descriptions = []
    
    count = 0
    # Process items in batches for more efficient I/O
    while True:
        data = output_queue.get()
        if data is None:  # Stop signal
            break
        
        # Unpack the data
        url, title, snippet, language, warc_id, content_type, server, warc_date, meta_desc = data
        
        # Super sanitize text fields
        title = sanitize_text(title)
        snippet = sanitize_text(snippet)
        meta_desc = sanitize_text(meta_desc)
        
        # Truncate very long fields
        title = title[:500] if title else ""
        snippet = snippet[:1000] if snippet else ""
        meta_desc = meta_desc[:500] if meta_desc else ""
        
        # Add to lists
        urls.append(url)
        titles.append(title)
        snippets.append(snippet)
        languages.append(language)
        warc_ids.append(warc_id)
        content_types.append(content_type)
        servers.append(server)
        warc_dates.append(warc_date)
        meta_descriptions.append(meta_desc)
        
        count += 1
        
        # Write in batches instead of all at once
        if count >= batch_size:
            # Write batch to parquet
            batch_df = pd.DataFrame({
                'url': urls,
                'title': titles,
                'snippet': snippets,
                'language': languages,
                'warc_id': warc_ids,
                'content_type': content_types,
                'server': servers,
                'warc_date': warc_dates,
                'meta_description': meta_descriptions
            })
            
            if append and os.path.exists(output_file):
                batch_df.to_parquet(f"temp_batch_{random.randint(1000, 9999)}.parquet", index=False)
            else:
                batch_df.to_parquet(output_file, index=False)
                append = True
                
            # Clear lists for next batch
            urls, titles, snippets = [], [], []
            languages, warc_ids, content_types = [], [], []
            servers, warc_dates, meta_descriptions = [], [], []
            count = 0
    
    # Create DataFrame from the accumulated data
    df = pd.DataFrame({
        'url': urls,
        'title': titles,
        'snippet': snippets,
        'language': languages,
        'warc_id': warc_ids,
        'warc_date': warc_dates,
        'meta_description': meta_descriptions
    })
    
    # Write to Parquet (append or write new)
    if append and os.path.exists(output_file):
        # Read existing file and append new data
        existing_df = pd.read_parquet(output_file)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_parquet(output_file, index=False)
    else:
        # Write new file
        df.to_parquet(output_file, index=False)

# Enhanced sanitize function to ensure CSV compatibility
def sanitize_text(text):
    if not text:
        return ""
    
    # Handle non-ASCII characters better by trying to translate common ones
    # before falling back to removal
    common_char_map = {
        '€': 'EUR', '£': 'GBP', '¥': 'JPY', '©': '(c)', '®': '(R)', '™': '(TM)',
        '•': '*', '…': '...', '—': '-', '–': '-', ''': "'", ''': "'", '"': '"', '"': '"',
        '«': '<<', '»': '>>', '¿': '?', '¡': '!', '°': ' degrees ',
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'ü': 'u', 'ñ': 'n', 'ç': 'c'
    }
    
    for char, replacement in common_char_map.items():
        text = text.replace(char, replacement)
    
    # Replace newlines, tabs and other whitespace with spaces
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    # Remove or simplify complex data structures like arrays and objects
    text = re.sub(r'Array\s*\([^)]*\)', '[array]', text)
    text = re.sub(r'\{[^}]*\}', '[object]', text)
    
    # Normalize unicode characters to their closest ASCII equivalents
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    
    # Remove any HTML tags that might remain
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove control characters
    text = ''.join(ch for ch in text if ord(ch) >= 32)
    
    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Escape any remaining problematic characters for CSV
    text = text.replace('"', '""')
    
    # Remove any other potentially problematic characters
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    
    # Truncate very long text more aggressively if needed
    if len(text) > 5000:
        text = text[:997] + "..." if text else ""
    
    return text

# Function to get list of WARC files from Common Crawl
def get_warc_files(path_listing_url, limit=None):
    """Download and parse a paths file to get WARC file URLs."""
    print(f"Downloading WARC file listing from {path_listing_url}")
    try:
        response = requests.get(path_listing_url, timeout=60)
        response.raise_for_status()
        
        urls = []
        with gzip.GzipFile(fileobj=response.raw) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                url = 'https://data.commoncrawl.org/' + line.decode('utf-8').strip()
                urls.append(url)
        
        print(f"Successfully parsed {len(urls)} WARC file URLs")
        return urls
    except Exception as e:
        print(f"Error accessing crawl data: {e}")
        return []

# Add new function to read local warc.paths file
def get_local_warc_files(paths_file, limit=None, random_selection=True, sample_size=None):
    """Get list of WARC files from local paths file"""
    with open(paths_file, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]
    
    if random_selection and sample_size:
        paths = random.sample(paths, min(sample_size, len(paths)))
    elif limit:
        paths = paths[:limit]
    
    return paths

# Function to save and load progress
def save_progress(state_file, processed_files):
    """Save progress to state file"""
    with open(state_file, 'w') as f:
        json.dump(list(processed_files), f)

def load_progress(state_file):
    """Load progress from state file"""
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            return set(json.load(f))
    return set()

# Create a download queue and worker pool
download_queue = Queue(maxsize=3)  # Buffer a few files ahead

def downloader(url_queue, download_queue):
    while True:
        url = url_queue.get()
        if url is None:
            break
        try:
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            download_queue.put((url, response.raw))
        except Exception as e:
            print(f"Download error {url}: {e}")
            download_queue.put((url, None))  # Signal error

# Main function
if __name__ == '__main__':
    # Configuration
    warc_paths_file = 'data/raw/warc.paths'  # Updated path
    output_file = 'data/processed/extracted_data.parquet'  # Updated path
    state_file = 'data/processed/extraction_state.json'  # Updated path
    max_per_domain = 100
    num_workers = 4
    batch_size = 1000
    max_files = 5000  # Significantly increased from 3 to process many more files
    max_records_per_file = 1000  # Keep this as is
    resume = True  # Whether to resume from previous run
    random_selection = True  # Whether to randomly select WARC files
    target_samples = 1000000  # Target number of samples
    
    # Add a counter for total samples extracted
    total_samples = 0
    if os.path.exists(output_file):
        # Use pandas to count rows in the parquet file
        total_samples = len(pd.read_parquet(output_file))
    print(f"Already have {total_samples} samples")
    
    # Load previously processed files if resuming
    processed_files = load_progress(state_file) if resume else set()
    print(f"Previously processed {len(processed_files)} files")

    # Get list of WARC files from local file with random selection
    warc_urls = get_local_warc_files(warc_paths_file, limit=max_files, random_selection=random_selection)
    print(f"Selected {len(warc_urls)} total WARC files")
    
    # Filter out already processed files
    if processed_files:
        warc_urls = [url for url in warc_urls if url not in processed_files]
    print(f"Will process {len(warc_urls)} new files")

    # Should we append to existing CSV?
    append_mode = resume and os.path.exists(output_file)
    
    # Process files
    for file_num, warc_url in enumerate(warc_urls):
        if total_samples >= target_samples:
            print(f"Target of {target_samples} samples reached!")
            break
            
        print(f"\nProcessing file {file_num+1}/{len(warc_urls)}: {os.path.basename(warc_url)}")
        print(f"Current sample count: {total_samples}/{target_samples}")
        
        # Set up multiprocessing
        input_queue = Queue(maxsize=100)
        output_queue = Queue()
        manager = Manager()
        domain_counts = manager.dict()
        lock = Lock()
        
        # Start worker processes
        workers = []
        for _ in range(num_workers):
            p = Process(target=worker, args=(input_queue, output_queue, domain_counts, lock, max_per_domain))
            p.start()
            workers.append(p)
        
        # Start writer process - append only if not the first file
        writer_p = Process(target=writer, args=(output_queue, output_file, append_mode or file_num > 0))
        writer_p.start()
        
        try:
            # Stream the WARC file
            print(f"Downloading: {warc_url}")
            response = requests.get(warc_url, stream=True, timeout=120)
            response.raise_for_status()
            
            # Process records as they stream in
            stream = response.raw
            record_count = 0
            for record in tqdm(ArchiveIterator(stream), desc="Processing records"):
                if record.rec_type == 'response':
                    try:
                        url = record.rec_headers.get_header('WARC-Target-URI')
                        warc_record_id = record.rec_headers.get_header('WARC-Record-ID')
                        warc_date = record.rec_headers.get_header('WARC-Date')
                        
                        # Extract HTTP headers
                        content_type = ""
                        server = ""
                        
                        # Read the HTTP headers
                        content = record.content_stream().read()
                        content_str = content.decode('utf-8', errors='replace')
                        
                        # Extract HTTP headers from the content
                        header_end = content_str.find('\r\n\r\n')
                        if header_end > 0:
                            headers_section = content_str[:header_end]
                            for line in headers_section.split('\r\n'):
                                if line.lower().startswith('content-type:'):
                                    content_type = line.split(':', 1)[1].strip()
                                elif line.lower().startswith('server:'):
                                    server = line.split(':', 1)[1].strip()
                        
                        # The HTML content starts after the headers
                        html = content_str[header_end+4:] if header_end > 0 else content_str
                        
                        # Send extracted data to workers
                        input_queue.put((url, html, warc_record_id, content_type, server, warc_date))
                        record_count += 1
                        if max_records_per_file and record_count >= max_records_per_file:
                            break
                    except Exception as e:
                        continue
                    
            # Signal workers to stop
            for _ in range(num_workers):
                input_queue.put(None)
            for p in workers:
                p.join()
            
            # Signal writer to stop
            output_queue.put(None)
            writer_p.join()
            
            # Update progress
            processed_files.add(warc_url)
            save_progress(state_file, processed_files)
            
            print(f"Finished file, processed {record_count} records")
            
        except Exception as e:
            print(f"Error processing {warc_url}: {e}")
            # Clean up processes in case of error
            for p in workers:
                if p.is_alive():
                    p.terminate()
            if writer_p.is_alive():
                writer_p.terminate()

        # After processing each file, update the sample count
        if os.path.exists(output_file):
            total_samples = len(pd.read_parquet(output_file))
        
        gc.collect()  # Force garbage collection

    print(f'Data extraction complete. Processed {len(processed_files)} files.')
    print(f'Results saved to {output_file}')