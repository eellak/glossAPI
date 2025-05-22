#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import asyncio
import aiohttp
import aiofiles
import argparse
import logging
import time
import re
from urllib.parse import urlparse, unquote
from datetime import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_indexes(keys):
    """Extract numeric prefixes from progress report keys."""
    nums = []
    for key in keys:
        try:
            # Try to extract the numeric part before the underscore
            match = re.match(r'paper_(\d+)_', key)
            if match:
                nums.append(int(match.group(1)))
        except (ValueError, TypeError):
            continue
    return nums

async def is_pdf(first_bytes):
    """Check if the file is a PDF by looking at its signature."""
    return first_bytes.startswith(b'%PDF')

async def download_file(session, url, output_path, req_type, filename, semaphore=None):
    """Download a file from the given URL."""
    if semaphore:
        async with semaphore:
            return await _download_file(session, url, output_path, req_type, filename)
    else:
        return await _download_file(session, url, output_path, req_type, filename)

async def _download_file(session, url, output_path, req_type, filename):
    """Internal function to perform the actual download."""
    file_path = os.path.join(output_path, filename)
    if os.path.exists(file_path):
        logger.info(f"File already exists, skipping: {filename}")
        return True, filename

    # URL needs to be properly encoded for non-ASCII characters
    parsed_url = urlparse(url)
    decoded_path = unquote(parsed_url.path)
    url = parsed_url._replace(path=decoded_path).geturl()

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept-Language': 'el-GR,el;q=0.9,en-US;q=0.8,en;q=0.7',
                'Connection': 'keep-alive'
            }

            if req_type.lower() == 'get':
                async with session.get(url, headers=headers, timeout=60) as response:
                    if response.status != 200:
                        logger.error(f"Failed to download {filename}, status code: {response.status}")
                        retry_count += 1
                        await asyncio.sleep(5)  # Wait a bit longer between retries
                        continue

                    # Read the first few bytes to check if it's a PDF
                    first_chunk = await response.content.read(10)
                    is_pdf_file = await is_pdf(first_chunk)
                    
                    if not is_pdf_file:
                        logger.error(f"Downloaded file is not a PDF: {filename}")
                        return False, filename

                    # Go back to the beginning of the content
                    response.content._cursor = 0
                    
                    try:
                        async with aiofiles.open(file_path, 'wb') as f:
                            # First write the chunk we already read
                            await f.write(first_chunk)
                            # Then continue with the rest
                            async for chunk in response.content.iter_chunked(8192):
                                await f.write(chunk)
                        logger.info(f"Downloaded {filename}")
                        return True, filename
                    except Exception as e:
                        logger.error(f"Failed to write file {filename}: {e}")
                        os.remove(file_path) if os.path.exists(file_path) else None
                        retry_count += 1
            else:
                logger.error(f"Request type {req_type} not supported")
                return False, filename
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"Error downloading {filename}: {e}")
            retry_count += 1
            await asyncio.sleep(5)  # Wait a bit longer between retries
        except Exception as e:
            logger.error(f"Unexpected error downloading {filename}: {e}")
            retry_count += 1
            await asyncio.sleep(5)
            
    return False, filename

async def download_batch(batch, output_path, progress_report, req_type, sleep_time, session, semaphore):
    """Download a batch of files."""
    logger.info(f"Starting to download batch of {len(batch)} files")
    tasks = []
    for item in batch:
        url = item['link']
        title = item.get('title', 'Unknown')
        idx = item.get('index', 0)
        
        # Create a filename based on the index and metadata
        filename = f"paper_{idx}_{title}.pdf"
        # Make sure the filename is valid for the filesystem
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Skip if already in progress report
        if filename in progress_report:
            logger.info(f"Skipping {filename} as it's already in progress report")
            continue
        
        logger.info(f"Queuing download for: {filename} from URL: {url}")
            
        # Add task to download the file
        task = asyncio.create_task(download_file(session, url, output_path, req_type, filename, semaphore))
        tasks.append((task, filename))
        
        # Sleep to avoid overwhelming the server
        await asyncio.sleep(sleep_time)
    
    if not tasks:
        logger.info("No files to download in this batch (all may be already downloaded)")
        return True
        
    logger.info(f"Waiting for {len(tasks)} download tasks to complete...")
    
    # Wait for all downloads to complete
    completed = 0
    for task, filename in tasks:
        try:
            success, _ = await task
            if success:
                progress_report[filename] = True
                completed += 1
                logger.info(f"Completed {completed}/{len(tasks)} downloads")
        except asyncio.CancelledError:
            logger.error(f"Task for {filename} was cancelled")
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
    
    logger.info(f"Batch completed: {completed}/{len(tasks)} files downloaded successfully")
    
    # Check if all files are downloaded
    return len(progress_report) >= len(batch)

async def run(args):
    """Main function to run the downloader."""
    # Load json file with sitemap
    try:
        with open(args.json, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
            # Convert dictionary to list of items with title and link
            data = []
            for title, link in data_dict.items():
                data.append({'title': title, 'link': link})
    except Exception as e:
        logger.error(f"Failed to load JSON file: {e}")
        return
        
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Path to progress report file
    progress_report_path = os.path.join(args.little_potato, 'progress_report.json')
    
    # Load progress report if it exists
    progress_report = {}
    try:
        if os.path.exists(progress_report_path):
            logger.info("Existing progress report found and loaded")
            with open(progress_report_path, 'r', encoding='utf-8') as f:
                progress_report = json.load(f)
        else:
            logger.info("No existing progress report found")
    except Exception as e:
        logger.error(f"Failed to load progress report: {e}")
    
    # Create a semaphore to limit concurrent downloads
    semaphore = asyncio.Semaphore(1)  # Only one concurrent download to be extra careful
    
    try:
        logger.info(f"Starting download from {args.json}")
        
        # Create a ClientSession
        timeout = aiohttp.ClientTimeout(total=300)  # 5-minute timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Process all sites
            all_downloaded = False
            items = []
            
            # Check if the JSON contains PDF links
            if args.type.lower() == 'pdf':
                # Add index to each item for tracking
                for i, item in enumerate(data):
                    if 'link' in item and 'pdf' in item['link'].lower():
                        item['index'] = i + 1
                        items.append(item)
                
                logger.info(f"Found {len(items)} PDF links to download")
            
            # If we already have progress, check which items we need to download
            if progress_report:
                try:
                    existing_indexes = get_indexes(list(progress_report.keys()))
                    # Filter out items we've already downloaded
                    items = [item for item in items if item['index'] not in existing_indexes]
                except Exception as e:
                    logger.error(f"An error occurred: {e}")
                    # Continue with all items if there was an error
            
            # Divide into batches if too many items
            batch_size = args.batch
            batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
            logger.info(f"Divided {len(items)} items into {len(batches)} batches of size {batch_size}")
            
            # Download batches
            finished = False
            for batch in batches:
                if not batch:  # Skip empty batches
                    continue
                    
                batch_finished = await download_batch(
                    batch, args.output, progress_report, args.req, args.sleep, session, semaphore
                )
                    
                # Save progress after each batch
                with open(progress_report_path, 'w', encoding='utf-8') as f:
                    json.dump(progress_report, f, ensure_ascii=False, indent=2)
                logger.info("Progress report written to progress_report.json")
                
                if batch_finished and not batches:
                    finished = True
            
            logger.info(f"Finished download from {args.json}")
            
        # Check if all files are downloaded
        if finished:
            logger.info("All PDF downloads completed!")
        else:
            logger.info("PDF downloads completed for this batch")
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    
    # Write progress report
    try:
        with open(progress_report_path, 'w', encoding='utf-8') as f:
            json.dump(progress_report, f, ensure_ascii=False, indent=2)
        logger.info("Progress report written to progress_report.json")
    except Exception as e:
        logger.error(f"Failed to write progress report: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download files from JSON sitemap.')
    parser.add_argument('--json', required=True, help='Path to JSON file with sitemap')
    parser.add_argument('--type', required=True, help='File type to download (e.g., pdf)')
    parser.add_argument('--req', required=True, help='Request type (get or post)')
    parser.add_argument('--output', required=True, help='Output directory path')
    parser.add_argument('--batch', type=int, default=5, help='Batch size (default: 5)')
    parser.add_argument('--sleep', type=float, default=3, help='Sleep time between requests in seconds (default: 3)')
    parser.add_argument('--little_potato', default=None, help='Directory to store progress_report.json (default: same as output)')
    
    args = parser.parse_args()
    
    # If little_potato is not specified, use output directory
    if args.little_potato is None:
        args.little_potato = args.output
        
    logger.info(f"Arguments received: JSON file: {args.json}, Sleep time: {args.sleep}, File type: {args.type}, Request type: {args.req}, Output path: {args.output}, 'progress_report.json' path: {args.little_potato}")
    
    # Run the async function
    asyncio.run(run(args))
