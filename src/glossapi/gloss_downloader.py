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
import json
import pandas as pd
from urllib.parse import urlparse, unquote
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

    def ensure_filename_base(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure a stable precomputed filename base column (without extension).

        - Creates and fills 'filename_base' for non-duplicate rows where missing.
        - Respects any pre-existing 'filename_base' values (does not recompute).
        - Uses AAA_000 style numbering in DataFrame order; avoids collisions.

        Args:
            df: DataFrame prepared for downloading (after expansion/dedup marking)

        Returns:
            pd.DataFrame with 'filename_base' populated where applicable
        """
        if 'filename_base' not in df.columns:
            df['filename_base'] = ""

        # Collect already assigned bases
        used: Set[str] = set(
            b for b in df.get('filename_base', pd.Series([], dtype=str)).astype(str).tolist() if b
        )

        counter = 0
        for idx in df.index:
            # Skip duplicates
            if 'is_duplicate' in df.columns:
                try:
                    if bool(df.at[idx, 'is_duplicate']):
                        continue
                except Exception:
                    pass
            # Skip rows without a valid URL
            try:
                uval = df.at[idx, self.url_column]
                if not isinstance(uval, str) or not uval.strip():
                    continue
            except Exception:
                pass

            base = str(df.at[idx, 'filename_base']) if 'filename_base' in df.columns else ''
            if base and base.strip():
                continue
            # Find next available base
            while True:
                cand = self.generate_filename(counter)
                counter += 1
                if cand not in used:
                    used.add(cand)
                    df.at[idx, 'filename_base'] = cand
                    break
        return df

    def precompute_filename_base_to_parquet(self, input_parquet: Union[str, Path], output_parquet: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Precompute 'filename_base' (AAA_000 style) and add it to a parquet file.

        - Expands list/JSON URL columns into one URL per row.
        - Marks duplicates (is_duplicate, duplicate_of) and assigns filename_base only to non-duplicates.
        - Respects existing filename_base values and does not recompute them.

        Args:
            input_parquet: Path to the input parquet to read and update
            output_parquet: Optional path to write the updated parquet; if None, overwrite input

        Returns:
            Updated DataFrame with filename_base
        """
        input_parquet = Path(input_parquet)
        df = pd.read_parquet(input_parquet)

        # Ensure URL column exists
        if self.url_column not in df.columns:
            available_columns = df.columns.tolist()
            self.logger.error(f"URL column '{self.url_column}' not found in parquet file. Available columns: {available_columns}")
            url_like_columns = [col for col in available_columns if any(keyword in col.lower() for keyword in ['url', 'link', 'uri'])]
            if url_like_columns:
                self.url_column = url_like_columns[0]
                self.logger.warning(f"Using '{self.url_column}' as URL column instead")
            else:
                raise ValueError(f"URL column '{self.url_column}' not found in parquet file and no alternative URL columns detected")

        # Reuse the same expansion + dedup logic
        df = self._expand_and_mark_duplicates(df)
        df = self.ensure_filename_base(df)

        # Drop internal columns not needed in saved parquet
        if '__url_norm' in df.columns:
            df = df.drop(columns=['__url_norm'])

        out_path = Path(output_parquet) if output_parquet else input_parquet
        df.to_parquet(out_path, index=False)
        self.logger.info(f"Precomputed filename_base and updated parquet at {out_path}")
        return df
    
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
        """Make a request with tenacity retry logic and return content, status, headers"""
        if requester == 'get':
            async with session.get(url, headers=headers, timeout=timeout) as response:
                response.raise_for_status()
                content = await response.read()
                # Capture headers before context closes
                resp_headers = dict(response.headers) if response.headers else {}
                return content, response.status, resp_headers
        else:  # post
            async with session.post(url, headers=headers, timeout=timeout) as response:
                response.raise_for_status()
                content = await response.read()
                resp_headers = dict(response.headers) if response.headers else {}
                return content, response.status, resp_headers

    def _ext_from_content_disposition(self, headers: Dict[str, str]) -> Optional[str]:
        """Try to extract file extension from Content-Disposition header"""
        if not headers:
            return None
        cd = None
        # Header names can vary in case; normalize
        for k, v in headers.items():
            if k.lower() == 'content-disposition':
                cd = v
                break
        if not cd:
            return None
        try:
            # Patterns: filename="name.ext" OR filename*=UTF-8''name.ext
            filename = None
            m = re.search(r'filename\*?=([^;]+)', cd, flags=re.IGNORECASE)
            if m:
                val = m.group(1).strip().strip('"')
                # Handle RFC 5987 encoding: UTF-8''...
                if "''" in val:
                    _, _, val = val.partition("''")
                filename = unquote(val)
            if not filename:
                return None
            ext = os.path.splitext(filename)[1]
            if ext:
                return ext.lstrip('.').lower()
        except Exception:
            return None
        return None

    def _ext_from_content_type(self, headers: Dict[str, str]) -> Optional[str]:
        """Map Content-Type header to a file extension"""
        if not headers:
            return None
        ctype = None
        for k, v in headers.items():
            if k.lower() == 'content-type':
                ctype = v
                break
        if not ctype:
            return None
        # Strip charset or params
        ctype = ctype.split(';', 1)[0].strip().lower()
        # Known precise mappings first
        known_map = {
            'application/pdf': 'pdf',
            'text/html': 'html',
            'application/xhtml+xml': 'html',
            'application/xml': 'xml',
            'text/xml': 'xml',
            'text/markdown': 'md',
            'text/csv': 'csv',
            'application/csv': 'csv',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
        }
        if ctype in known_map:
            return known_map[ctype]
        try:
            ext = mimetypes.guess_extension(ctype)  # returns like '.pdf'
            if ext:
                return ext.lstrip('.').lower()
        except Exception:
            return None
        return None

    def _ext_from_magic_bytes(self, content: bytes) -> Optional[str]:
        """Infer file type from leading bytes when headers/URL are insufficient"""
        if not content:
            return None
        head = content[:4096]
        # PDF
        if head.startswith(b'%PDF-'):
            return 'pdf'
        # HTML (very simple heuristic)
        lower_head = head.lower()
        if b'<!doctype html' in lower_head or b'<html' in lower_head:
            return 'html'
        # XML
        if lower_head.lstrip().startswith(b'<?xml'):
            return 'xml'
        # OOXML/ZIP containers (docx/pptx/xlsx)
        if head[:4] in (b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08'):
            try:
                import zipfile, io
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    names = {n.lower() for n in zf.namelist()}
                    if any(n.startswith('word/') for n in names):
                        return 'docx'
                    if any(n.startswith('ppt/') for n in names):
                        return 'pptx'
                    # Not supported types (e.g., xlsx) are ignored here
            except Exception:
                pass
        return None

    def infer_file_extension(self, url: str, headers: Dict[str, str], content: bytes) -> str:
        """Infer the most likely file extension using URL, headers and content bytes"""
        # 1) URL path extension
        url_ext = self.get_file_extension_from_url(url)
        if self.is_supported_format(url_ext):
            return url_ext

        # 2) Content-Disposition filename
        cd_ext = self._ext_from_content_disposition(headers)
        if cd_ext and self.is_supported_format(cd_ext):
            return cd_ext

        # 3) Content-Type header
        ct_ext = self._ext_from_content_type(headers)
        # Normalize common synonyms
        if ct_ext in { 'htm', 'xhtml' }:
            ct_ext = 'html'
        if ct_ext and self.is_supported_format(ct_ext):
            return ct_ext

        # 4) Magic byte sniffing
        sniff_ext = self._ext_from_magic_bytes(content)
        if sniff_ext and self.is_supported_format(sniff_ext):
            return sniff_ext

        # 5) Fall back to URL ext if any, otherwise 'bin'
        return url_ext if url_ext else 'bin'
    
    async def download_file(self, row_index: int, url: str, semaphore: asyncio.Semaphore, 
                           rate_limiter: RateLimiter, retry_count: int = 0,
                           filename_base: Optional[str] = None) -> Tuple[bool, str, str, str, int]:
        """
        Download a file from a URL
        
        Args:
            row_index: Index in the dataframe
            url: URL to download
            semaphore: Semaphore for concurrency control
            rate_limiter: Rate limiter for API limits
            retry_count: Current retry count
        Returns:
            Tuple[bool, str, str, str, int]: (success, filename, file_ext, error_message, retry_count)
        """
        if not url or pd.isna(url):
            return False, "", "", "Empty URL", retry_count
        
        # Get a new user-agent for each request
        user_agent = next(self.user_agents)
        domain = urlparse(url).netloc
        
        # Ensure URL has scheme
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        
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
                            content, status, resp_headers = await self.make_request(
                                session, requester, url, headers, timeout
                            )
                            
                            self.verbose_log(f"Request successful with status {status}")
                            
                            # Infer file extension using URL, response headers and content
                            file_ext = self.infer_file_extension(url, resp_headers, content)
                            if not self.is_supported_format(file_ext):
                                self.logger.warning(f"Unsupported file format after inference: {file_ext}. Supported formats: {', '.join(self.supported_formats)}")
                                return False, "", file_ext or "", f"Unsupported file format: {file_ext}", retry_count
                            # Decide final filename using precomputed base if available
                            if filename_base and str(filename_base).strip():
                                filename = f"{filename_base}.{file_ext}"
                            else:
                                filename = self.generate_filename(row_index, file_ext)
                            
                            # Write to file
                            await self.write_file(filename, content, self.downloads_dir)
                            
                            self.logger.info(f"Successfully downloaded {filename} from {url}")
                            return True, filename, file_ext, "", retry_count
                            
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
                            # Best-effort ext from URL if possible
                            try:
                                url_ext = self.get_file_extension_from_url(url)
                            except Exception:
                                url_ext = ""
                            return False, "", url_ext, error_msg, retry_count + 1
                            
                        except Exception as e:
                            error_msg = str(e)
                            self.logger.error(f"Failed to download {url}: {error_msg}")
                            
                            # Detailed verbose logging for general errors
                            if self.verbose:
                                self.logger.debug(f"Exception type: {type(e).__name__}")
                                self.logger.debug(f"Exception details: {e}")
                                import traceback
                                self.logger.debug(f"Traceback: {traceback.format_exc()}")
                            
                            try:
                                url_ext = self.get_file_extension_from_url(url)
                            except Exception:
                                url_ext = ""
                            return False, "", url_ext, error_msg, retry_count + 1
                            
                    except asyncio.TimeoutError:
                        self.logger.error(f"Overall timeout exceeded for {url}")
                        return False, "", "", "Timeout", retry_count + 1
                    except Exception as e:
                        self.logger.error(f"Error downloading {url}: {str(e)}")
                        return False, "", "", str(e), retry_count + 1
                        
            except aiohttp.ClientError as e:
                error_msg = str(e)
                self.logger.error(f"ClientError while downloading {url}: {error_msg}")
                try:
                    url_ext = self.get_file_extension_from_url(url)
                except Exception:
                    url_ext = ""
                return False, "", url_ext, error_msg, retry_count + 1
            except asyncio.TimeoutError:
                self.logger.error(f"Timeout while downloading {url}")
                return False, "", "", "Timeout", retry_count + 1
            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"Error while downloading {url}: {error_msg}")
                try:
                    url_ext = self.get_file_extension_from_url(url)
                except Exception:
                    url_ext = ""
                return False, "", url_ext, error_msg, retry_count + 1

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
        # New: store inferred file extension
        if 'file_ext' not in df.columns:
            df['file_ext'] = ""
        # Duplicate reporting and traceability columns
        if 'is_duplicate' not in df.columns:
            df['is_duplicate'] = False
        if 'duplicate_of' not in df.columns:
            df['duplicate_of'] = ""
        if 'source_row' not in df.columns:
            # Default to the current index if not provided
            try:
                df['source_row'] = df.index.astype('int64')
            except Exception:
                df['source_row'] = 0
        if 'url_index' not in df.columns:
            df['url_index'] = 0
        
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

                # Skip duplicates identified during preprocessing
                if 'is_duplicate' in df.columns:
                    try:
                        if bool(df.loc[row_idx, 'is_duplicate']):
                            self.logger.info(f"Skipping duplicate URL at row {row_idx}")
                            continue
                    except Exception:
                        pass

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
                        retry_count=retry_count,
                        filename_base=(df.loc[row_idx, 'filename_base'] if 'filename_base' in df.columns else None)
                    )
                )
                tasks.append((row_idx, task))
            
            # Process tasks as they complete
            for row_idx, task in tasks:
                try:
                    success, filename, file_ext, error, retry_count = await task
                    
                    # Update DataFrame with results
                    df.loc[row_idx, 'download_success'] = success
                    df.loc[row_idx, 'filename'] = filename
                    df.loc[row_idx, 'file_ext'] = file_ext
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
    
    def download_files(self, input_parquet: str, **kwargs) -> pd.DataFrame:
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
        
        # Prepare (expand + dedup mark) and precompute filename_base
        df = self._expand_and_mark_duplicates(df)
        df = self.ensure_filename_base(df)

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
        # Drop internal normalization column before returning
        if '__url_norm' in updated_df.columns:
            try:
                updated_df = updated_df.drop(columns=['__url_norm'])
            except Exception:
                pass

        # Summary of download results
        success_count = updated_df['download_success'].sum()
        fail_count = len(updated_df) - success_count
        self.logger.info(f"Download complete: {success_count} successful, {fail_count} failed, files downloaded to {self.downloads_dir}")
        
        return updated_df

    def _expand_and_mark_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expand list/JSON URL cells to one URL per row and mark duplicates.

        Adds columns: source_row, url_index, is_duplicate, duplicate_of, __url_norm.
        Initializes download-related columns when expanding.
        """
        # If URL column contains lists or JSON arrays, expand to one URL per row
        def _parse_links(value: Any) -> List[str]:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return []
            # Already a list/tuple of URLs or dicts
            if isinstance(value, (list, tuple)):
                out: List[str] = []
                for v in value:
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        continue
                    if isinstance(v, dict):
                        u = v.get('url') or v.get('href') or v.get('link')
                        if u:
                            out.append(str(u))
                    else:
                        out.append(str(v))
                return out
            # Dict with a url field
            if isinstance(value, dict):
                u = value.get('url') or value.get('href') or value.get('link')
                return [str(u)] if u else []
            # String content: could be a JSON array or JSON object or plain URL
            if isinstance(value, str):
                s = value.strip()
                # JSON list
                if (s.startswith('[') and s.endswith(']')) or (s.startswith('{') and s.endswith('}')):
                    try:
                        obj = json.loads(s)
                        return _parse_links(obj)
                    except Exception:
                        pass
                # Fallback: single URL string
                return [s]
            # Unknown types -> string cast
            try:
                return [str(value)]
            except Exception:
                return []

        sample_vals = df[self.url_column].dropna().head(5).tolist()
        needs_expand = any(
            isinstance(v, (list, tuple)) or (isinstance(v, str) and v.strip().startswith('['))
            for v in sample_vals
        )

        # Track original row ids for traceability when expanding
        if needs_expand:
            records = []
            for orig_idx, row in df.iterrows():
                links = _parse_links(row.get(self.url_column))
                if not links:
                    # Keep a placeholder row with empty URL so failures are recorded
                    new_row = row.to_dict()
                    new_row[self.url_column] = ''
                    new_row['source_row'] = int(orig_idx)
                    new_row['url_index'] = -1
                    records.append(new_row)
                    continue
                for pos, u in enumerate(links):
                    new_row = row.to_dict()
                    new_row[self.url_column] = u
                    new_row['source_row'] = int(orig_idx)
                    new_row['url_index'] = int(pos)
                    records.append(new_row)
            df = pd.DataFrame.from_records(records)
            # Reset index to ensure unique sequential indices for filename generation
            df = df.reset_index(drop=True)
            # Reset downloader columns to clean defaults if present
            df['download_success'] = False
            df['filename'] = ""
            df['download_error'] = ""
            df['download_retry_count'] = 0
            df['file_ext'] = ""
            df['is_duplicate'] = False
            df['duplicate_of'] = ""
        else:
            # Ensure traceability columns exist even without expansion
            if 'source_row' not in df.columns:
                df['source_row'] = df.index.astype('int64')
            if 'url_index' not in df.columns:
                df['url_index'] = 0
            if 'is_duplicate' not in df.columns:
                df['is_duplicate'] = False
            if 'duplicate_of' not in df.columns:
                df['duplicate_of'] = ""
            if 'download_error' not in df.columns:
                df['download_error'] = ""
            if 'download_success' not in df.columns:
                df['download_success'] = False

        # Normalize URLs and mark duplicates (keep first occurrence)
        from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

        def _canon(u: str) -> str:
            if not isinstance(u, str) or not u:
                return ''
            try:
                if not u.startswith(('http://', 'https://')):
                    u = 'https://' + u
                p = urlparse(u)
                scheme = p.scheme.lower()
                netloc = p.netloc.lower()
                path = re.sub('/+', '/', p.path)
                # Keep query but sort keys for stable dedup
                qs = urlencode(sorted(parse_qsl(p.query, keep_blank_values=True)), doseq=True)
                return urlunparse((scheme, netloc, path, '', qs, ''))
            except Exception:
                return str(u)

        df['__url_norm'] = df[self.url_column].map(_canon)
        dup_mask = df['__url_norm'].duplicated(keep='first')
        canon_rows = df.loc[~dup_mask, ['__url_norm', self.url_column]]
        canon_map = dict(zip(canon_rows['__url_norm'], canon_rows[self.url_column]))
        df.loc[dup_mask, 'is_duplicate'] = True
        df.loc[dup_mask, 'duplicate_of'] = df.loc[dup_mask, '__url_norm'].map(canon_map).fillna('')
        df.loc[dup_mask, 'download_error'] = df.loc[dup_mask, 'download_error'].mask(
            df.loc[dup_mask, 'download_error'].astype(str).str.len() == 0,
            'Duplicate URL (skipped)'
        )

        return df
