#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concurrent Downloader

A versatile concurrent downloader that uses asyncio and aiohttp to efficiently download
files from URLs. It accepts parquet files with URLs and metadata columns, downloads the files
concurrently, and creates unique filenames with a structured naming pattern.

Features:
- Parquet file integration for metadata handling
- Unique filename generation with the pattern paper_AAA000, paper_AAA001, etc.
- Configurable concurrency
- Retry mechanism for failed downloads
- Download status tracking
- Works with any file type
"""

import aiohttp
import asyncio
import os
import argparse
import time
import random
import logging
import re
import string
import aiofiles
import pandas as pd
from urllib.parse import urlparse
from collections import deque
from typing import Dict, List, Tuple, Set, Optional, Any, Iterator
import mimetypes
import string
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_exponential, retry_if_exception_type, retry_if_result, before_sleep_log
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("concurrent_download.log")
    ]
)
logger = logging.getLogger(__name__)

# Configure tenacity logger
tenacity_logger = logging.getLogger('tenacity')
tenacity_logger.setLevel(logging.INFO)

# Add specific loggers for libraries that can be noisy
logging.getLogger('aiohttp').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

# Rate limiter class for API limits
class RateLimiter:
    """Rate limiter to enforce API rate limits"""
    
    def __init__(self, rate_limit: int, time_period: int = 60):
        """
        Initialize rate limiter
        
        Args:
            rate_limit: Maximum number of requests allowed in time_period
            time_period: Time period in seconds (default: 60 seconds = 1 minute)
        """
        self.rate_limit = rate_limit
        self.time_period = time_period
        self.request_timestamps = deque(maxlen=rate_limit)
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """
        Acquire permission to make a request, waiting if necessary
        """
        async with self.lock:
            current_time = time.time()
            
            # If we haven't reached the limit yet, allow immediately
            if len(self.request_timestamps) < self.rate_limit:
                self.request_timestamps.append(current_time)
                return
            
            # Check if the oldest request is outside the time window
            elapsed = current_time - self.request_timestamps[0]
            if elapsed < self.time_period:
                # We need to wait until a slot is available
                wait_time = self.time_period - elapsed
                logger.debug(f"Rate limit reached. Waiting {wait_time:.2f} seconds")
                # Release the lock while waiting
                await asyncio.sleep(wait_time)
                # Reacquire and check again (recursive call)
                await self.acquire()
            else:
                # We can make a request now
                self.request_timestamps.popleft()  # Remove oldest
                self.request_timestamps.append(current_time)

# Constants for filename generation
LETTERS = string.ascii_uppercase
DIGITS = string.digits


def generate_filename(index: int, file_ext: str = None) -> str:
    """
    Generate a filename in the format AAA_000, AAA_001, etc.
    
    Args:
        index: Sequential number to convert to the AAA_000 format
        file_ext: Optional file extension (with dot)
        
    Returns:
        str: Unique filename
    """
    # Calculate letter part (AAA, AAB, etc.)
    letter_base = ord('A') # ASCII code for 'A'
    first_letter = chr(letter_base + (index // (26*26)) % 26)
    second_letter = chr(letter_base + (index // 26) % 26)
    third_letter = chr(letter_base + index % 26)
    
    # Calculate number part (000, 001, etc.)
    number_part = f"{(index % 1000):03d}"
    
    letters = f"{first_letter}{second_letter}{third_letter}"
    digits = number_part
    
    if file_ext:
        return f"{letters}_{digits}.{file_ext}"
    else:
        return f"{letters}_{digits}"


def get_file_extension_from_url(url: str) -> str:
    """
    Extract file extension from URL or guess based on content type
    
    Args:
        url: URL to extract extension from
        
    Returns:
        str: File extension (without dot)
    """
    # First try to get extension from URL path
    path = urlparse(url).path
    ext = os.path.splitext(path)[1].lower()
    
    if ext and ext.startswith('.'):
        return ext[1:]  # Remove the leading dot
    
    # If no extension found, return a default
    return "bin"


def get_mime_type(url: str) -> str:
    """
    Get MIME type from URL
    
    Args:
        url: URL to get MIME type for
        
    Returns:
        str: MIME type
    """
    mime_type, _ = mimetypes.guess_type(url)
    return mime_type if mime_type else "application/octet-stream"


async def get_base_url(url: str) -> str:
    """
    Extract base URL from a full URL
    
    Args:
        url: Full URL
        
    Returns:
        str: Base URL (scheme + netloc)
    """
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return base_url


async def setup_session(session: aiohttp.ClientSession, url: str, headers: Dict[str, str]) -> Dict[str, str]:
    """
    Initialize the session with base headers
    
    Args:
        session: aiohttp ClientSession
        url: URL to access
        headers: Headers to use
        
    Returns:
        Dict[str, str]: Updated headers
    """
    base_url = await get_base_url(url)
    initial_url = base_url
    try:
        async with session.get(initial_url, headers=headers, timeout=10) as response:
            await response.text()
    except Exception as e:
        logger.warning(f"Failed to setup session for {base_url}: {e}")
    return headers


async def write_file(filename: str, content: bytes, output_path: str = "./") -> str:
    """
    Write downloaded content to a file
    
    Args:
        filename: Name of the file
        content: Binary content to write
        output_path: Directory to write to
        
    Returns:
        str: Path to the written file
    """
    path_to_file = os.path.join(output_path, filename)
    async with aiofiles.open(path_to_file, 'wb') as file:
        await file.write(content)
    return path_to_file


def user_agent_generator() -> Iterator[str]:
    """
    Generate random user-agents to avoid bot detection
    
    Yields:
        str: Random user agent string
    """
    templates = [
        "Mozilla/5.0 ({os}) AppleWebKit/537.36 (KHTML, like Gecko) {browser}/{version} Safari/537.36",
        "Mozilla/5.0 ({os}) Gecko/20100101 {browser}/{version}",
        "Mozilla/5.0 ({os}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36 {browser}/{alt_version}"
    ]
    operating_systems = [
        "Windows NT 10.0; Win64; x64",
        "Macintosh; Intel Mac OS X 10_15_7",
        "X11; Linux x86_64",
        "Windows NT 6.1; Win64; x64",
        "Android 12; Mobile"
    ]
    browsers = [
        ("Chrome", random.randint(90, 110), "Chrome"),
        ("Firefox", random.randint(90, 110), "Firefox"),
        ("Edge", random.randint(90, 110), "Edg"),
        ("Safari", random.randint(600, 610), "Safari")
    ]
    while True:
        template = random.choice(templates)
        os_name = random.choice(operating_systems)
        browser, version, alt_browser = random.choice(browsers)
        full_version = f"{version}.0.{random.randint(1000, 9999)}"
        alt_version = f"{random.randint(90, 110)}.0.{random.randint(1000, 9999)}"
        user_agent = template.format(os=os_name, browser=browser, version=full_version, alt_browser=alt_browser, alt_version=alt_version)
        yield user_agent


@retry(stop=(stop_after_attempt(3) | stop_after_delay(30)),
       wait=wait_exponential(multiplier=1, min=2, max=10),
       retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
       reraise=True,
       before_sleep=before_sleep_log(tenacity_logger, logging.INFO))
async def make_request(session, requester, url, headers, timeout):
    """Make a request with tenacity retry logic"""
    async with requester(
        url, 
        headers=headers, 
        allow_redirects=True,
        max_redirects=10,
        verify_ssl=False,
        timeout=timeout
    ) as response:
        content = None
        if response.status == 200:
            content = await response.read()
        return response.status, content

async def download_file(row_index: int, url: str, semaphore: asyncio.Semaphore, 
                      args: argparse.Namespace, user_agent: str, rate_limiter: RateLimiter,
                      retry_count: int = 0) -> Tuple[bool, str, str, int]:
    """
    Download a file from a URL
    
    Args:
        row_index: Index in the dataframe
        url: URL to download
        semaphore: Semaphore for concurrency control
        args: Command-line arguments
        user_agent: User agent to use
        retry_count: Current retry count
        
    Returns:
        Tuple[bool, str, str, int]: (success, filename, error_message, retry_count)
    """
    # Skip empty URLs
    if pd.isna(url) or not url:
        return (False, "", "Empty URL", retry_count + 1)
    
    # Get base URL for referer
    base_url = await get_base_url(url)
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    
    # Ensure URL has scheme
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    
    # Get file extension from URL
    file_ext = get_file_extension_from_url(url)
    
    # Generate unique filename
    filename = generate_filename(row_index, file_ext)
    
    # Enhanced headers with common browser-like attributes to bypass 403 errors
    headers = {
        'User-Agent': user_agent,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'cross-site',
        'Pragma': 'no-cache',
        'Cache-Control': 'no-cache',
        'TE': 'trailers',
        'Referer': f"https://www.google.com/search?q={domain}",
        'Origin': base_url,
        'DNT': '1'
    }
    
    # Add cookie handling if needed for specific domains
    cookies = {}
    if 'europarl.europa.eu' in url or 'data.europarl.europa.eu' in url:
        cookies = {
            'cookie_consent': 'accepted',
            'ec_cookiepopin': 'NjY1ODJjNDg5NDc1ODlkNzYwZDA0OTU5NzJkYWI2ZTc',
            'JSESSIONID': f"session-id-{random.randint(100000000, 999999999)}",
            'loadedEP': 'true',
            'GUEST_LANGUAGE_ID': 'en_US'
        }
    
    async with semaphore:
        # Implement exponential backoff
        sleep_time = args.sleep * (2 ** retry_count)
        await asyncio.sleep(random.uniform(sleep_time, sleep_time * 1.5))
        
        # Set up timeout with exponential backoff
        timeout = aiohttp.ClientTimeout(total=60 + (retry_count * 15))
        
        try:
            # Acquire permission from rate limiter before making request
            await rate_limiter.acquire()
            
            # Create session with proper connection pooling
            conn = aiohttp.TCPConnector(
                ssl=False,
                limit_per_host=2,  # Limit concurrent connections per host
                force_close=False,  # Keep connections open for reuse
                enable_cleanup_closed=True
            )
            
            async with aiohttp.ClientSession(
                connector=conn,
                timeout=timeout,
                trust_env=True,  # Use environment for proxy information
                cookies=cookies  # Use our cookies
            ) as session:
                # Try to access the base domain first to establish cookies
                if retry_count == 0:  # Only do this on first attempt
                    try:
                        # Get permission from rate limiter for the base URL request
                        await rate_limiter.acquire()
                        
                        async with session.get(
                            base_url, 
                            headers=headers,
                            allow_redirects=True,
                            timeout=aiohttp.ClientTimeout(total=15)
                        ) as response:
                            await response.read()
                            await asyncio.sleep(random.uniform(1.0, 2.0))
                    except Exception as e:
                        logger.debug(f"Initial base URL visit failed: {str(e)}")
                
                # Determine request method (get or post)
                request_method = args.request_method.lower()
                requester = getattr(session, request_method)
                
                # Attempt the download with tenacity-powered retry logic
                try:
                    # Use tenacity retry wrapper for the actual request
                    status, content = await asyncio.wait_for(
                        make_request(session, requester, url, headers, timeout),
                        timeout=args.request_timeout  # Overall timeout for the whole operation
                    )
                    
                    if status == 200 and content:
                        await write_file(filename, content, args.output_dir)
                        logger.info(f"Successfully downloaded {filename} from {url}")
                        return (True, filename, "", retry_count)
                    elif status in [403, 429]:
                        # Special handling for 403/429 (Forbidden/Too Many Requests)
                        await asyncio.sleep(random.uniform(3.0, 5.0))  # Longer wait
                        logger.warning(f"Received {status} for {url}")
                        error_msg = f"HTTP {status}"
                        return (False, filename, error_msg, retry_count + 1)
                    else:
                        error_msg = f"HTTP {status}"
                        logger.error(f"Failed to download {url}: {error_msg}")
                        return (False, filename, error_msg, retry_count + 1)
                                
                except asyncio.TimeoutError:
                    logger.error(f"Overall timeout exceeded for {url}")
                    return (False, filename, "Request timed out", retry_count + 1)
                except Exception as e:
                    logger.error(f"Error downloading {url}: {str(e)}")
                    return (False, filename, f"Download error: {str(e)}", retry_count + 1)
                        
        except aiohttp.ClientError as e:
            error_msg = f"Client error: {str(e)}"
            logger.error(f"ClientError while downloading {url}: {error_msg}")
        except asyncio.TimeoutError:
            error_msg = "Timeout error"
            logger.error(f"Timeout while downloading {url}")
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"Error while downloading {url}: {error_msg}")
            
        return (False, filename, error_msg, retry_count + 1)


async def download_files(df: pd.DataFrame, url_column: str, semaphore: asyncio.Semaphore, 
                        args: argparse.Namespace, rate_limiter: RateLimiter,
                        max_retries: int = 3) -> pd.DataFrame:
    """
    Download files from URLs in a DataFrame using internal batching for memory efficiency
    
    Args:
        df: DataFrame with URLs
        url_column: Name of the column containing URLs
        semaphore: Semaphore for concurrency control
        args: Command-line arguments
        max_retries: Maximum number of retries per URL
        
    Returns:
        pd.DataFrame: Updated DataFrame with download results
    """
    # Add columns for filenames and download status if they don't exist
    if 'filename' not in df.columns:
        df['filename'] = None
    if 'download_success' not in df.columns:
        df['download_success'] = False
    if 'error_message' not in df.columns:
        df['error_message'] = ""
    
    # Create a user agent generator
    user_agent_gen = user_agent_generator()
    
    # Calculate output parquet path (needed for periodic saves)
    output_parquet = os.path.join(args.output_dir, os.path.basename(args.input_parquet))
    if args.output_parquet:
        output_parquet = args.output_parquet
    
    # Get total number of unprocessed rows
    unprocessed_mask = pd.isna(df['download_success']) | ~df['download_success']
    unprocessed_indices = df[unprocessed_mask].index.tolist()
    total_unprocessed = len(unprocessed_indices)
    
    logger.info(f"Found {total_unprocessed} unprocessed rows out of {len(df)} total")
    
    internal_batch_size = args.internal_batch_size
    successful_downloads = 0
    periodic_save_count = args.save_every
    
    # Process in batches to save memory
    for batch_start in range(0, total_unprocessed, internal_batch_size):
        batch_end = min(batch_start + internal_batch_size, total_unprocessed)
        current_batch_indices = unprocessed_indices[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//internal_batch_size + 1} of {(total_unprocessed + internal_batch_size - 1)//internal_batch_size}: rows {batch_start} to {batch_end-1}")
        
        # Create tasks for current batch
        tasks = []
        for row_idx in current_batch_indices:
            url = df.at[row_idx, url_column]
            # Get the retry count from the dataframe if it exists
            retry_count = int(df.at[row_idx, 'retry_count']) if 'retry_count' in df.columns and pd.notna(df.at[row_idx, 'retry_count']) else 0
            
            # Skip URLs that have failed too many times
            if args.skip_failed_after > 0 and retry_count >= args.skip_failed_after:
                logger.info(f"Skipping URL at row {row_idx} - too many failures: {retry_count}")
                continue
                
            if pd.notna(url):
                task = asyncio.create_task(
                    download_file(
                        row_idx, url, semaphore, args, 
                        next(user_agent_gen), rate_limiter, retry_count
                    )
                )
                tasks.append((row_idx, task))
        
        # Process tasks in current batch
        for row_idx, task in tasks:
            try:
                success, filename, error_msg, new_retry_count = await task
                df.at[row_idx, 'filename'] = filename
                df.at[row_idx, 'download_success'] = success
                df.at[row_idx, 'error_message'] = error_msg
                df.at[row_idx, 'retry_count'] = new_retry_count
                
                # Count successful downloads and save periodically
                if success:
                    successful_downloads += 1
                    if successful_downloads % periodic_save_count == 0:
                        logger.info(f"Periodic save: Completed {successful_downloads} downloads. Saving progress to {output_parquet}")
                        df.to_parquet(output_parquet, index=False)
                        
            except Exception as e:
                logger.error(f"Error processing task for row {row_idx}: {e}")
                df.at[row_idx, 'download_success'] = False
                df.at[row_idx, 'error_message'] = f"Task error: {str(e)}"
        
        # Save after each batch
        logger.info(f"Batch complete. Saving progress to {output_parquet}")
        df.to_parquet(output_parquet, index=False)
    
    return df


async def run(args: argparse.Namespace) -> None:
    """
    Main function to run the downloader
    
    Args:
        args: Command-line arguments
    """
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine output parquet path
    output_parquet = os.path.join(args.output_dir, os.path.basename(args.input_parquet))
    if args.output_parquet:
        output_parquet = args.output_parquet
    
    # Check if we're resuming from a previous run
    resuming = False
    if os.path.exists(output_parquet) and args.resume:
        try:
            logger.info(f"Found existing output parquet file at {output_parquet}. Attempting to resume.")
            df = pd.read_parquet(output_parquet)
            resuming = True
            
            # Count successful downloads for statistics
            existing_success_count = df['download_success'].sum() if 'download_success' in df.columns else 0
            logger.info(f"Resuming from previous run with {existing_success_count} already completed downloads")
            
        except Exception as e:
            logger.warning(f"Failed to read existing parquet for resuming: {e}. Starting fresh.")
            resuming = False
    
    # If not resuming, read the input parquet
    if not resuming:
        logger.info(f"Reading input parquet file: {args.input_parquet}")
        df = pd.read_parquet(args.input_parquet)
    
    original_count = len(df)
    logger.info(f"Loaded {original_count} rows from parquet file")
    
    # Check if URL column exists
    if args.url_column not in df.columns:
        raise ValueError(f"URL column '{args.url_column}' not found in parquet file. Available columns: {', '.join(df.columns)}")
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(args.concurrency)
    
    # Create rate limiter (100 requests per minute)
    rate_limiter = RateLimiter(args.rate_limit, args.rate_period)
    logger.info(f"Using rate limit of {args.rate_limit} requests per {args.rate_period} seconds")
    
    # Process downloads
    logger.info(f"Starting downloads with concurrency: {args.concurrency}")
    updated_df = await download_files(df, args.url_column, semaphore, args, rate_limiter, args.max_retries)
    
    # Save updated DataFrame to parquet
    logger.info(f"Saving updated parquet file to: {output_parquet}")
    updated_df.to_parquet(output_parquet, index=False)
    
    # Report statistics
    success_count = updated_df['download_success'].sum() if 'download_success' in updated_df.columns else 0
    logger.info(f"Download summary:")
    logger.info(f"  Total URLs: {original_count}")
    logger.info(f"  Successfully downloaded: {success_count}")
    logger.info(f"  Failed: {original_count - success_count}")
    logger.info(f"Updated parquet file saved to: {output_parquet}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Concurrent downloader for files from a parquet file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input-parquet', required=True,
                        help='Path to the input parquet file')
    parser.add_argument('--url-column', required=True,
                        help='Column name containing URLs in the parquet file')
    parser.add_argument('--output-dir', default='./downloads',
                        help='Directory to save downloaded files')
    parser.add_argument('--output-parquet',
                        help='Path to save the updated parquet file')
    parser.add_argument('--internal-batch-size', type=int, default=100,
                        help='Number of files to process in one internal batch (for memory efficiency)')
    parser.add_argument('--save-every', type=int, default=50,
                        help='Save progress to parquet file every N successful downloads')
    parser.add_argument('--concurrency', type=int, default=5,
                        help='Number of concurrent downloads')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Maximum retry attempts for failed downloads')
    parser.add_argument('--sleep', type=float, default=0.5,
                        help='Base sleep time between requests in seconds')
    parser.add_argument('--request-method', choices=['get', 'post'], default='get',
                       help='HTTP request method to use')
    parser.add_argument('--resume', action='store_true',
                       help='Resume downloading from a previously saved checkpoint')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--retry-interval', type=float, default=5.0,
                       help='Time to wait between retries for 403/429 errors (seconds)')
    parser.add_argument('--rate-limit', type=int, default=100,
                       help='Maximum number of requests per time period (rate limiting)')
    parser.add_argument('--rate-period', type=int, default=60,
                       help='Time period in seconds for rate limiting')
    parser.add_argument('--request-timeout', type=int, default=45,
                       help='Overall timeout in seconds for each request')
    parser.add_argument('--skip-failed-after', type=int, default=3,
                       help='Skip URLs that failed more than this many times')
    
    return parser.parse_args()


async def main() -> None:
    """
    Main entry point
    """
    args = parse_args()
    try:
        await run(args)
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
