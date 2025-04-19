#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GlossDownloader

A versatile concurrent downloader for the GlossAPI pipeline that uses asyncio and aiohttp 
to efficiently download files from URLs. It processes parquet files with URL columns,
downloads the files concurrently, and creates unique filenames.

Features:
- Parquet file integration for metadata handling
- Unique filename generation with structured naming pattern
- Configurable concurrency
- Retry mechanism for failed downloads
- Download status tracking
- Rate limiting support
- Domain-specific cookie handling
"""

import aiohttp
import asyncio
import os
import time
import random
import logging
import re
import string
import aiofiles
import pandas as pd
from urllib.parse import urlparse
from collections import deque
from typing import Dict, List, Tuple, Set, Optional, Any, Iterator, Union
import mimetypes
from pathlib import Path
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_exponential, retry_if_exception_type, before_sleep_log
import functools


class RateLimiter:
    """Rate limiter to enforce API rate limits with robust handling of timeouts and interruptions"""
    
    def __init__(self, rate_limit: int, time_period: int = 60, max_wait_time: int = 30):
        """
        Initialize rate limiter
        
        Args:
            rate_limit: Maximum number of requests allowed in time_period
            time_period: Time period in seconds (default: 60 seconds = 1 minute)
            max_wait_time: Maximum time to wait for rate limiting in seconds (default: 30)
        """
        self.rate_limit = rate_limit
        self.time_period = time_period
        self.max_wait_time = max_wait_time
        self.request_timestamps = deque(maxlen=rate_limit)
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
    
    async def acquire(self):
        """
        Acquire permission to make a request, waiting if necessary.
        Includes safeguards against excessive waiting and better handling of cancellations.
        """
        try:
            async with self.lock:
                current_time = time.time()
                
                # If we haven't reached the limit yet, allow immediately
                if len(self.request_timestamps) < self.rate_limit:
                    self.request_timestamps.append(current_time)
                    return
                
                # Check if the oldest request is outside the time window
                elapsed = current_time - self.request_timestamps[0]
                if elapsed < self.time_period:
                    # Calculate wait time but cap it to max_wait_time
                    wait_time = min(self.time_period - elapsed, self.max_wait_time)
                    self.logger.debug(f"Rate limit reached. Waiting {wait_time:.2f} seconds (capped at {self.max_wait_time}s)")
                    
                    # Create a cancellable wait task
                    try:
                        # Release the lock while waiting
                        self.lock.release()
                        await asyncio.wait_for(asyncio.sleep(wait_time), timeout=wait_time+1)
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Rate limit wait timed out after {wait_time} seconds")
                    except asyncio.CancelledError:
                        self.logger.warning("Rate limit wait was cancelled")
                        raise  # Re-raise to properly handle cancellation
                    finally:
                        # Make sure to reacquire the lock
                        await self.lock.acquire()
                    
                    # Clean old timestamps that might have expired during our wait
                    self._clean_expired_timestamps(current_time)
                    
                    # Add our timestamp now
                    self.request_timestamps.append(time.time())
                else:
                    # We can make a request now
                    self.request_timestamps.popleft()  # Remove oldest
                    self.request_timestamps.append(current_time)
        except Exception as e:
            self.logger.error(f"Error in rate limiter: {e}")
            # If we encounter any issue, allow the request to proceed
            # Better to occasionally exceed rate limits than to hang indefinitely
            return
    
    def _clean_expired_timestamps(self, current_time):
        """Remove any expired timestamps from the queue"""
        while self.request_timestamps and (current_time - self.request_timestamps[0] >= self.time_period):
            self.request_timestamps.popleft()


class GlossDownloader:
    """
    A concurrent downloader for fetching files from URLs using asyncio and aiohttp.
    Downloads files from URLs in a Parquet file and organizes them with unique filenames.
    """
    
    def __init__(
        self,
        url_column: str = 'url',
        output_dir: str = './downloads',
        internal_batch_size: int = 100,
        save_every: int = 50,
        concurrency: int = 5,
        max_retries: int = 3,
        sleep: float = 0.5,
        request_method: str = 'get',
        retry_interval: float = 5.0,
        rate_limit: int = 100,
        rate_period: int = 60,
        request_timeout: int = 45,
        skip_failed_after: int = 3,
        domain_cookies: Dict[str, Dict[str, str]] = None,
        supported_formats: List[str] = None,
        log_level: int = logging.INFO,
        verbose: bool = False
    ):
        """
        Initialize the downloader with configuration parameters.
        
        Args:
            url_column: Column name containing URLs in the parquet file
            output_dir: Directory to save downloaded files
            internal_batch_size: Number of files in one internal batch for memory efficiency
            save_every: Save progress to parquet file every N successful downloads
            concurrency: Number of concurrent downloads
            max_retries: Maximum retry attempts for failed downloads
            sleep: Base sleep time between requests in seconds
            request_method: HTTP request method ('get' or 'post')
            retry_interval: Time to wait between retries for 403/429 errors (seconds)
            rate_limit: Maximum number of requests per time period (rate limiting)
            rate_period: Time period in seconds for rate limiting
            request_timeout: Overall timeout in seconds for each request
            skip_failed_after: Skip URLs that failed more than this many times
            domain_cookies: Dictionary mapping domain names to cookies dictionaries
            log_level: Logging level (default: logging.INFO)
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Verbose logging flag
        self.verbose = verbose
        
        # Define a verbose log function to be used throughout the class
        def verbose_log(self, message, level=logging.DEBUG):
            if self.verbose:
                self.logger.log(level, message)
        
        # Bind the verbose_log method to this instance
        self.verbose_log = functools.partial(verbose_log, self)
        
        # Store configuration parameters
        self.url_column = url_column
        self.output_dir = Path(output_dir)
        self.internal_batch_size = internal_batch_size
        self.save_every = save_every
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.sleep = sleep
        self.request_method = request_method
        self.retry_interval = retry_interval
        # Initialize rate limiter if rate_limit is specified
        self.rate_limiter = None
        if rate_limit > 0:
            # Use more conservative settings for rate limiting to avoid hanging
            # Default max_wait_time of 30 seconds prevents indefinite waiting
            self.rate_limiter = RateLimiter(
                rate_limit=rate_limit, 
                time_period=rate_period,
                max_wait_time=30  # Cap maximum wait time to 30 seconds
            )
        self.request_timeout = request_timeout
        self.skip_failed_after = skip_failed_after
        
        # Setup domain cookies with defaults
        self.domain_cookies = domain_cookies or {}
        
        # Set supported formats for validation
        self.supported_formats = supported_formats or ['pdf', 'docx', 'xml', 'html', 'pptx', 'csv', 'md']
        
        # Create downloads directory inside output directory
        self.downloads_dir = self.output_dir / "downloads"
        os.makedirs(self.downloads_dir, exist_ok=True)
        
        # Constants for filename generation
        self.LETTERS = string.ascii_uppercase
        self.DIGITS = string.digits
        
        # Set up user agent generator
        self.user_agents = self._user_agent_generator()
    
    def generate_filename(self, index: int, file_ext: str = None) -> str:
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
    
    def get_file_extension_from_url(self, url: str) -> str:
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
    
    def _user_agent_generator(self) -> Iterator[str]:
        """
        Generate random user-agents to avoid bot detection
        
        Yields:
            str: Random user agent string
        """
        # List of common browsers and their user agent strings
        chrome_version = f"{random.randint(70, 100)}.0.{random.randint(1000, 9999)}.{random.randint(10, 999)}"
        firefox_version = f"{random.randint(60, 90)}.0"
        safari_version = f"{random.randint(10, 15)}.{random.randint(0, 9)}.{random.randint(1, 9)}"
        edge_version = f"{random.randint(80, 100)}.0.{random.randint(100, 999)}.{random.randint(10, 99)}"
        
        user_agents = [
            f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_version} Safari/537.36",
            f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_{random.randint(1, 7)}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_version} Safari/537.36",
            f"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:{firefox_version}) Gecko/20100101 Firefox/{firefox_version}",
            f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_{random.randint(1, 7)}) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{safari_version} Safari/605.1.15",
            f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_version} Edge/{edge_version}",
            f"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_version} Safari/537.36",
        ]
        
        while True:
            # Shuffle and yield each user agent
            random.shuffle(user_agents)
            for agent in user_agents:
                yield agent
    
    async def get_base_url(self, url: str) -> str:
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
    
    async def setup_session(self, session: aiohttp.ClientSession, url: str, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Initialize the session with base headers
        
        Args:
            session: aiohttp ClientSession
            url: URL to access
            headers: Headers to use
            
        Returns:
            Dict[str, str]: Updated headers
        """
        base_url = await self.get_base_url(url)
        initial_url = base_url
        try:
            # Access the base domain to get cookies
            async with session.get(initial_url, headers=headers, timeout=10):
                pass
            return headers
        except Exception as e:
            self.logger.warning(f"Failed to setup session for {base_url}: {e}")
            return headers
    
    async def write_file(self, filename: str, content: bytes, output_path: Union[str, Path]) -> str:
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
        
    def is_supported_format(self, file_ext: str) -> bool:
        """
        Check if the file extension is in the list of supported formats
        
        Args:
            file_ext: File extension without the dot (e.g., 'pdf')
            
        Returns:
            bool: True if supported, False otherwise
        """
        return file_ext.lower() in self.supported_formats
    
    @retry(
        stop=(stop_after_attempt(3) | stop_after_delay(30)),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=(retry_if_exception_type(aiohttp.ClientError) | 
               retry_if_exception_type(asyncio.TimeoutError)),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.INFO)
    )
    async def make_request(self, session, requester, url, headers, timeout):
        """Make a request with tenacity retry logic"""
        if requester == 'get':
            async with session.get(url, headers=headers, timeout=timeout) as response:
                response.raise_for_status()
                content = await response.read()
                return content, response.status
        else:  # post
            async with session.post(url, headers=headers, timeout=timeout) as response:
                response.raise_for_status()
                content = await response.read()
                return content, response.status
    
    async def download_file(self, row_index: int, url: str, semaphore: asyncio.Semaphore, 
                           rate_limiter: RateLimiter, retry_count: int = 0) -> Tuple[bool, str, str, int]:
        """
        Download a file from a URL
        
        Args:
            row_index: Index in the dataframe
            url: URL to download
            semaphore: Semaphore for concurrency control
            rate_limiter: Rate limiter for API limits
            retry_count: Current retry count
            
        Returns:
            Tuple[bool, str, str, int]: (success, filename, error_message, retry_count)
        """
        if not url or pd.isna(url):
            return False, "", "Empty URL", retry_count
        
        # Get a new user-agent for each request
        user_agent = next(self.user_agents)
        domain = urlparse(url).netloc
        
        # Ensure URL has scheme
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        
        # Get file extension from URL
        file_ext = self.get_file_extension_from_url(url)
        
        # Generate unique filename
        filename = self.generate_filename(row_index, file_ext)
        
        # Get base URL for referer header
        base_url = await self.get_base_url(url)
        
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
        
        # Check for domain-specific cookies
        cookies = {}
        for domain_pattern, domain_cookies in self.domain_cookies.items():
            if domain_pattern in url:
                cookies.update(domain_cookies)
                # If the domain needs dynamic values like random IDs
                for key, value in cookies.items():
                    if 'random.randint' in str(value):
                        # Replace with an actual random value (only supporting this pattern for now)
                        if 'session-id' in value:
                            cookies[key] = f"session-id-{random.randint(100000000, 999999999)}"
        
        async with semaphore:
            # Apply rate limiting
            await rate_limiter.acquire()
            
            # Implement exponential backoff
            sleep_time = self.sleep * (2 ** retry_count)
            await asyncio.sleep(random.uniform(sleep_time, sleep_time * 1.5))
            
            # Set up timeout with exponential backoff
            timeout = aiohttp.ClientTimeout(
                total=min(self.request_timeout * (1.5 ** retry_count), 180),  # Cap at 3 minutes
                connect=min(30 * (1.2 ** retry_count), 60),  # Cap connect timeout at 1 minute
                sock_connect=min(30 * (1.2 ** retry_count), 60),  # Cap socket connect at 1 minute
                sock_read=min(60 * (1.2 ** retry_count), 120)  # Cap socket read at 2 minutes
            )
            
            try:
                # Create a new session for each download to avoid cookie contamination
                async with aiohttp.ClientSession(cookies=cookies) as session:
                    try:
                        # Try to access the base domain first to establish cookies
                        headers = await self.setup_session(session, url, headers)
                        
                        # Set a shorter timeout for the initial connection attempt
                        base_timeout = aiohttp.ClientTimeout(total=10)
                        try:
                            # Visit the base domain to establish cookies if needed
                            base_domain = urlparse(url).netloc
                            if any(domain in base_domain for domain in self.domain_cookies.keys()):
                                base_url = f"https://{base_domain}"
                                async with session.get(base_url, headers=headers, timeout=base_timeout):
                                    pass
                        except Exception as e:
                            # Non-fatal error, just log and continue
                            self.logger.debug(f"Initial base URL visit failed: {str(e)}")
                            pass
                        
                        # Choose request method
                        requester = self.request_method.lower()
                        
                        try:
                            # Verbose logging of request attempt
                            self.verbose_log(f"Attempting download request to URL: {url}")
                            self.verbose_log(f"Request method: {requester}, Timeout: {timeout.total}s")
                            self.verbose_log(f"Headers: {headers}")
                            
                            # Attempt the download with tenacity-powered retry logic
                            content, status = await self.make_request(
                                session, requester, url, headers, timeout
                            )
                            
                            self.verbose_log(f"Request successful with status {status}")
                            
                            # Check if file extension is supported
                            file_ext = self.get_file_extension_from_url(url)
                            if not self.is_supported_format(file_ext):
                                self.logger.warning(f"Unsupported file format: {file_ext}. Supported formats: {', '.join(self.supported_formats)}")
                                return False, "", f"Unsupported file format: {file_ext}", retry_count
                                
                            # Write to file
                            await self.write_file(filename, content, self.downloads_dir)
                            
                            self.logger.info(f"Successfully downloaded {filename} from {url}")
                            return True, filename, "", retry_count
                            
                        except aiohttp.ClientResponseError as e:
                            # Handle HTTP errors
                            status = e.status
                            self.logger.warning(f"Received {status} for {url}")
                            
                            # Detailed verbose logging for HTTP errors
                            if self.verbose:
                                self.logger.debug(f"HTTP Error Details - Status: {e.status}, Message: {e.message}")
                                self.logger.debug(f"Headers: {e.headers if hasattr(e, 'headers') else 'No headers available'}")
                                self.logger.debug(f"Request info: {e.request_info if hasattr(e, 'request_info') else 'No request info available'}")
                            
                            error_msg = f"HTTP {status}: {str(e)}"
                            return False, "", error_msg, retry_count + 1
                            
                        except Exception as e:
                            error_msg = str(e)
                            self.logger.error(f"Failed to download {url}: {error_msg}")
                            
                            # Detailed verbose logging for general errors
                            if self.verbose:
                                self.logger.debug(f"Exception type: {type(e).__name__}")
                                self.logger.debug(f"Exception details: {e}")
                                import traceback
                                self.logger.debug(f"Traceback: {traceback.format_exc()}")
                            
                            return False, "", error_msg, retry_count + 1
                            
                    except asyncio.TimeoutError:
                        self.logger.error(f"Overall timeout exceeded for {url}")
                        return False, "", "Timeout", retry_count + 1
                    except Exception as e:
                        self.logger.error(f"Error downloading {url}: {str(e)}")
                        return False, "", str(e), retry_count + 1
                        
            except aiohttp.ClientError as e:
                error_msg = str(e)
                self.logger.error(f"ClientError while downloading {url}: {error_msg}")
                return False, "", error_msg, retry_count + 1
            except asyncio.TimeoutError:
                self.logger.error(f"Timeout while downloading {url}")
                return False, "", "Timeout", retry_count + 1
            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"Error while downloading {url}: {error_msg}")
                return False, "", error_msg, retry_count + 1

    async def _download_files_async(self, df: pd.DataFrame, semaphore: asyncio.Semaphore, 
                                   rate_limiter: RateLimiter) -> pd.DataFrame:
        """
        Core async function to download files from URLs in a DataFrame
        
        Args:
            df: DataFrame with URLs
            semaphore: Semaphore for concurrency control
            rate_limiter: Rate limiter for API throttling
            
        Returns:
            pd.DataFrame: Updated DataFrame with download results
        """
        # Create a copy we can modify
        df = df.copy()
        
        # Initialize result columns if they don't exist
        if 'download_success' not in df.columns:
            df['download_success'] = False
        if 'filename' not in df.columns:
            df['filename'] = ""
        if 'download_error' not in df.columns:
            df['download_error'] = ""
        if 'download_retry_count' not in df.columns:
            df['download_retry_count'] = 0
        
        # Get total unprocessed rows (not downloaded successfully)
        mask = ~df['download_success']
        total_unprocessed = mask.sum()
        
        # Get indices for rows that need processing
        unprocessed_indices = df.index[mask].tolist()
        
        self.logger.info(f"Found {total_unprocessed} unprocessed rows out of {len(df)} total")
        
        # Variables to track progress
        successful_downloads = 0
        output_parquet = None
        
        # Process in internal batches for memory efficiency
        for batch_start in range(0, total_unprocessed, self.internal_batch_size):
            batch_end = min(batch_start + self.internal_batch_size, total_unprocessed)
            batch_indices = unprocessed_indices[batch_start:batch_end]
            
            self.logger.info(f"Processing batch {batch_start//self.internal_batch_size + 1} of {(total_unprocessed + self.internal_batch_size - 1)//self.internal_batch_size}: rows {batch_start} to {batch_end-1}")
            
            # Create tasks for batch
            tasks = []
            for i, row_idx in enumerate(batch_indices):
                url = df.loc[row_idx, self.url_column]
                retry_count = df.loc[row_idx, 'download_retry_count']
                
                # Skip URLs that have failed too many times
                if retry_count >= self.skip_failed_after:
                    self.logger.info(f"Skipping URL at row {row_idx} - too many failures: {retry_count}")
                    continue
                
                task = asyncio.create_task(
                    self.download_file(
                        row_index=row_idx,
                        url=url,
                        semaphore=semaphore,
                        rate_limiter=rate_limiter,
                        retry_count=retry_count
                    )
                )
                tasks.append((row_idx, task))
            
            # Process tasks as they complete
            for row_idx, task in tasks:
                try:
                    success, filename, error, retry_count = await task
                    
                    # Update DataFrame with results
                    df.loc[row_idx, 'download_success'] = success
                    df.loc[row_idx, 'filename'] = filename
                    df.loc[row_idx, 'download_error'] = error
                    df.loc[row_idx, 'download_retry_count'] = retry_count
                    
                    if success:
                        successful_downloads += 1
                        
                        # Periodically save progress
                        if successful_downloads % self.save_every == 0:
                            self.logger.info(f"Periodic save: Completed {successful_downloads} downloads.")
                    
                except Exception as e:
                    self.logger.error(f"Error processing task for row {row_idx}: {e}")
                    df.loc[row_idx, 'download_error'] = str(e)
                    df.loc[row_idx, 'download_retry_count'] += 1
        
        return df
    
    def download_files(self, input_parquet: str) -> pd.DataFrame:
        """
        Download files from URLs in a parquet file
        
        Args:
            input_parquet: Path to input parquet file
            
        Returns:
            pd.DataFrame: DataFrame with download results
        """
        self.logger.info(f"Loading parquet file: {input_parquet}")
        
        # Load input parquet
        df = pd.read_parquet(input_parquet)
        
        # Check that URL column exists
        if self.url_column not in df.columns:
            available_columns = df.columns.tolist()
            self.logger.error(f"URL column '{self.url_column}' not found in parquet file. Available columns: {available_columns}")
            # Try to find a column that might contain URLs
            url_like_columns = [col for col in available_columns if any(keyword in col.lower() for keyword in ['url', 'link', 'uri'])]
            if url_like_columns:
                self.url_column = url_like_columns[0]
                self.logger.warning(f"Using '{self.url_column}' as URL column instead")
            else:
                raise ValueError(f"URL column '{self.url_column}' not found in parquet file and no alternative URL columns detected")
        
        # Ensure downloads directory exists
        os.makedirs(self.downloads_dir, exist_ok=True)
        
        # Initialize semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.concurrency)
        
        # Use the already configured rate limiter if available
        rate_limiter = self.rate_limiter
        if not rate_limiter:
            # Create a minimal rate limiter if none was configured
            rate_limiter = RateLimiter(100, 60, 30)  # Default: 100 requests per minute, max 30s wait
        
        # Run async download
        self.logger.info(f"Starting download with concurrency={self.concurrency}, rate_limit={rate_limiter.rate_limit}/{rate_limiter.time_period}s")
        
        # Run the async pipeline
        loop = asyncio.get_event_loop()
        updated_df = loop.run_until_complete(
            self._download_files_async(df, semaphore, rate_limiter)
        )
        
        # Summary of download results
        success_count = updated_df['download_success'].sum()
        fail_count = len(updated_df) - success_count
        self.logger.info(f"Download complete: {success_count} successful, {fail_count} failed, files downloaded to {self.downloads_dir}")
        
        return updated_df
