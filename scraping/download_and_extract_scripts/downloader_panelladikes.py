#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Downloader script for Panhellenic (Panelladikes) Exams PDFs
This script downloads PDF files containing exam questions and solutions
from various years of the Panhellenic Exams in Greece.
"""

import os
import json
import time
import logging
import argparse
import asyncio
import aiohttp
import aiofiles
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'panelladikes_downloader.log'), 'a')
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

async def download_file(session, url, title, output_dir, semaphore):
    """
    Downloads a single file asynchronously

    Args:
        session (aiohttp.ClientSession): HTTP session
        url (str): URL to download
        title (str): Title/description of the file to use for naming
        output_dir (str): Directory to save the file
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent downloads

    Returns:
        dict: Dictionary with result information
    """
    async with semaphore:
        # Replace invalid characters in title with underscores
        safe_title = "".join([c if c.isalnum() or c in [' ', '.', '-', '_'] else '_' for c in title])
        safe_title = safe_title.replace(' ', '_').replace('>', '_')

        # Create a filename from the title (limited to 100 chars to avoid too long filenames)
        filename = f"{safe_title[:100]}.pdf"
        filepath = os.path.join(output_dir, filename)

        # Don't re-download if file exists
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            logger.info(f"Skipping {url}: File already exists")
            return {"title": title, "url": url, "status": "already_exists", "filename": filename}

        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Set up headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf,*/*',
            'Accept-Language': 'en-US,en;q=0.9,el;q=0.8',
            'Connection': 'keep-alive',
        }

        try:
            # Download the file with retries
            for attempt in range(3):  # retry up to 3 times
                try:
                    logger.info(f"Downloading {url} (Attempt {attempt+1}/3)")
                    async with session.get(url, headers=headers, timeout=60) as response:
                        if response.status != 200:
                            logger.warning(f"Failed to download {url}: HTTP {response.status}")
                            await asyncio.sleep(2 * (attempt + 1))  # Exponential backoff
                            continue

                        # Download the file in chunks to avoid loading entire file in memory
                        data = await response.read()

                        # Check if it's a valid PDF (starts with %PDF)
                        if data[:4] != b'%PDF':
                            logger.warning(f"Not a valid PDF file: {url}")
                            return {"title": title, "url": url, "status": "not_valid_pdf", "filename": filename}

                        # Save to file
                        async with aiofiles.open(filepath, 'wb') as f:
                            await f.write(data)

                        logger.info(f"Downloaded {url} to {filepath}")
                        return {"title": title, "url": url, "status": "success", "filename": filename}

                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(f"Error downloading {url} (Attempt {attempt+1}/3): {str(e)}")
                    await asyncio.sleep(2 * (attempt + 1))  # Exponential backoff

            # If we get here, all retries failed
            logger.error(f"Failed to download {url} after 3 attempts")
            return {"title": title, "url": url, "status": "failed", "filename": filename}

        except Exception as e:
            logger.exception(f"Unexpected error downloading {url}: {str(e)}")
            return {"title": title, "url": url, "status": "error", "filename": filename, "error": str(e)}


async def download_batch(session, data_batch, output_dir, sleep_time, progress_file):
    """
    Downloads a batch of files

    Args:
        session (aiohttp.ClientSession): HTTP session
        data_batch (list): List of dictionaries with title and url
        output_dir (str): Directory to save the files
        sleep_time (int): Time to sleep between requests
        progress_file (str): File to save progress

    Returns:
        list: List of dictionaries with result information
    """
    logger.info(f"Starting batch download of {len(data_batch)} files")

    # Limit to 1 concurrent download to be gentle on the server
    semaphore = asyncio.Semaphore(1)

    results = []
    for idx, item in enumerate(data_batch):
        title = item["title"]
        url = item["link"]

        # Sleep between requests to avoid overwhelming the server
        if idx > 0:
            await asyncio.sleep(sleep_time)

        result = await download_file(session, url, title, output_dir, semaphore)
        results.append(result)

        # Save progress after each file
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results


async def download_from_json(json_file, output_dir, batch_size, sleep_time, progress_dir):
    """
    Downloads files from a JSON file

    Args:
        json_file (str): Path to the JSON file
        output_dir (str): Directory to save the files
        batch_size (int): Number of files to download in each batch
        sleep_time (int): Time to sleep between requests
        progress_dir (str): Directory to save progress files
    """
    logger.info(f"Arguments received: JSON file: {json_file}, Output dir: {output_dir}, "
                f"Batch size: {batch_size}, Sleep time: {sleep_time}, Progress dir: {progress_dir}")

    # Create output and progress directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(progress_dir, exist_ok=True)

    # Load the JSON file (handle both a list and a dict format)
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Normalize the data to a list of dicts with title and link
        items = []
        if isinstance(data, dict):
            # Handle dict format
            logger.info(f"JSON file contains a dictionary with {len(data)} items")
            for title, link in data.items():
                # Fix common URL issues
                if "eduadvisor.grhttp" in link:
                    # Fix incorrectly merged URLs
                    fixed_link = link.replace("eduadvisor.grhttp", "http")
                    logger.info(f"Fixed malformed URL for {title}: {fixed_link}")
                    link = fixed_link

                # Skip known broken URLs
                if (("ΠΑΝΕΛΛΗΝΙΕΣ%202013/ΘΕΜΑΤΑ%20ΚΑΙ%20ΑΠΑΝΤΗΣΕΙΣ/2013/" in link) or
                    ("ΠΑΝΕΛΛΗΝΙΕΣ%202013/ΘΕΜΑΤΑ%20ΚΑΙ%20ΑΠΑΝΤΗΣΕΙΣ/2011/" in link)):
                    logger.warning(f"Skipping known problematic URL for {title}: {link}")
                    continue

                items.append({"title": title, "link": link})
        elif isinstance(data, list):
            # Handle list format
            logger.info(f"JSON file contains a list with {len(data)} items")
            for item in data:
                link = item.get("link", "")
                title = item.get("title", "")

                if "eduadvisor.grhttp" in link:
                    # Fix incorrectly merged URLs
                    fixed_link = link.replace("eduadvisor.grhttp", "http")
                    logger.info(f"Fixed malformed URL for {title}: {fixed_link}")
                    item["link"] = fixed_link

                # Skip known broken URLs
                if (("ΠΑΝΕΛΛΗΝΙΕΣ%202013/ΘΕΜΑΤΑ%20ΚΑΙ%20ΑΠΑΝΤΗΣΕΙΣ/2013/" in link) or
                    ("ΠΑΝΕΛΛΗΝΙΕΣ%202013/ΘΕΜΑΤΑ%20ΚΑΙ%20ΑΠΑΝΤΗΣΕΙΣ/2011/" in link)):
                    logger.warning(f"Skipping known problematic URL for {title}: {link}")
                    continue

                items.append(item)
        else:
            raise ValueError(f"Unexpected JSON format: {type(data)}")

        logger.info(f"Loaded {len(items)} items from {json_file}")

        # Create a progress file
        progress_file = os.path.join(progress_dir, 'progress_report.json')
        downloaded = []

        # Load existing progress if available
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    downloaded = json.load(f)
                logger.info(f"Loaded progress: {len(downloaded)} items already downloaded")
            except json.JSONDecodeError:
                logger.warning(f"Could not parse progress file {progress_file}, starting from scratch")

        # Filter out already downloaded items
        downloaded_urls = {item["url"] for item in downloaded if item["status"] in ["success", "already_exists"]}
        items_to_download = [item for item in items if item["link"] not in downloaded_urls]

        logger.info(f"Found {len(items_to_download)} new items to download")

        # Prepare batches
        batches = [items_to_download[i:i+batch_size] for i in range(0, len(items_to_download), batch_size)]
        logger.info(f"Split into {len(batches)} batches of up to {batch_size} items each")

        # Download files in batches
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_idx+1}/{len(batches)}")

            # Create a new session for each batch
            async with aiohttp.ClientSession() as session:
                new_results = await download_batch(session, batch, output_dir, sleep_time, progress_file)
                downloaded.extend(new_results)

                # Save overall progress
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump(downloaded, f, ensure_ascii=False, indent=2)

            # Sleep between batches
            if batch_idx < len(batches) - 1:
                logger.info(f"Sleeping for {sleep_time} seconds before next batch")
                await asyncio.sleep(sleep_time)

        # Final summary
        success_count = sum(1 for item in downloaded if item["status"] == "success")
        already_exists_count = sum(1 for item in downloaded if item["status"] == "already_exists")
        failed_count = sum(1 for item in downloaded if item["status"] in ["failed", "error", "not_valid_pdf"])

        logger.info(f"Download complete. Summary: {success_count} successful, {already_exists_count} already existed, {failed_count} failed")

    except Exception as e:
        logger.exception(f"Error processing JSON file {json_file}: {str(e)}")
        raise


def main():
    """Main function to handle command line arguments and start the download process"""
    parser = argparse.ArgumentParser(description='Download PDFs from a JSON file')
    parser.add_argument('--json', required=True, help='Path to the JSON file')
    parser.add_argument('--type', default='pdf', help='Type of files to download (default: pdf)')
    parser.add_argument('--req', default='get', help='HTTP request type (get/post) (default: get)')
    parser.add_argument('--output', required=True, help='Output directory for downloaded files')
    parser.add_argument('--batch', type=int, default=5, help='Batch size (default: 5)')
    parser.add_argument('--sleep', type=int, default=3, help='Sleep time in seconds between requests (default: 3)')
    parser.add_argument('--little_potato', default='', help='Progress directory (default: same as output)')

    args = parser.parse_args()

    # If progress directory is not specified, use the output directory
    progress_dir = args.little_potato or args.output

    # Start the download process
    asyncio.run(download_from_json(
        args.json,
        args.output,
        args.batch,
        args.sleep,
        progress_dir
    ))


if __name__ == "__main__":
    main()
