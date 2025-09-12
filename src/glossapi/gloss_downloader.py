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
from dataclasses import dataclass, field
import mimetypes
from pathlib import Path
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_exponential, retry_if_exception_type, before_sleep_log
import functools
from email.utils import parsedate_to_datetime


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
        """Acquire permission to make a request, waiting if necessary."""
        while True:
            async with self.lock:
                now = time.time()
                # purge expired within the window
                while self.request_timestamps and (now - self.request_timestamps[0] >= self.time_period):
                    self.request_timestamps.popleft()
                if len(self.request_timestamps) < self.rate_limit:
                    self.request_timestamps.append(now)
                    return
                # compute delay without holding lock
                next_allowed = self.request_timestamps[0] + self.time_period
                delay = min(max(0.0, next_allowed - now), self.max_wait_time)
            await asyncio.sleep(delay)
    
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
        verbose: bool = False,
        # New scheduler controls (backwards-compatible defaults)
        scheduler_mode: str = 'global',  # 'global' (existing) | 'per_domain'
        scheduler_group_by: str = 'base_domain',  # 'base_domain' or a DataFrame column to group by (e.g., 'collection_slug')
        per_domain_concurrency: int = 5,  # start each domain with up to 5 parallel requests
        max_active_domains: Optional[int] = None,  # defaults based on global concurrency
        eta_max_seconds: int = 4 * 24 * 3600,  # 4 days
        dynamic_tuning: bool = True,
        domain_concurrency_floor: int = 1,
        domain_concurrency_ceiling: Optional[int] = None,
        # High-level progress logging
        progress_log_file: Optional[Union[str, Path]] = None,
        request_log_file: Optional[Union[str, Path]] = None,
        progress_log_level: int = logging.INFO,
        # TLS/SSL controls
        ssl_verify: bool = True,
        ssl_cafile: Optional[str] = None,
        # Per-row referer support: name of a column in the input parquet that
        # contains the page URL where the file link was found
        referer_column: Optional[str] = None,
        # Checkpointing
        checkpoint_every: Optional[int] = 500,  # write checkpoint parquet every N completed tasks
        checkpoint_seconds: Optional[float] = 60.0,  # or every N seconds (whichever triggers first)
        # Domain availability probing
        pre_ping_domains: bool = True,
        ping_timeout_s: float = 5.0,
        ping_concurrency: int = 20,
        ping_method: str = 'head',  # 'head' or 'get'
        ping_recheck_seconds: float = 60.0,
        down_wait_max_seconds: float = 300.0,
        timeout_streak_threshold: int = 5,
        backoff_min_s: float = 60.0,
        backoff_max_s: float = 900.0,
        error_burst_window: int = 20,
        error_burst_threshold: float = 0.5,
        park_403_seconds: float = 600.0,
        _used_filename_bases: Optional[Set[str]] = None,
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
        
        # Create standard directories inside output directory
        self.downloads_dir = self.output_dir / "downloads"
        os.makedirs(self.downloads_dir, exist_ok=True)
        self.logs_dir = self.output_dir / "logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Constants for filename generation
        self.LETTERS = string.ascii_uppercase
        self.DIGITS = string.digits
        
        # Set up user agent generator
        self.user_agents = self._user_agent_generator()
        # Used filename bases to prevent collisions on resume
        self._used_filename_bases: Set[str] = set(_used_filename_bases or [])

        # Scheduler controls
        self.scheduler_mode = scheduler_mode
        self.per_domain_concurrency = max(1, int(per_domain_concurrency))
        self.scheduler_group_by = str(scheduler_group_by)
        self.max_active_domains = max_active_domains  # may be None to auto-compute
        self.eta_max_seconds = int(eta_max_seconds)
        self.dynamic_tuning = bool(dynamic_tuning)
        self.domain_concurrency_floor = max(1, int(domain_concurrency_floor))
        # Default ceiling: start value (per_domain_concurrency); user may raise if desired
        self.domain_concurrency_ceiling = (
            int(domain_concurrency_ceiling)
            if domain_concurrency_ceiling is not None
            else max(1, int(per_domain_concurrency))
        )
        # Checkpoint cadence
        self.checkpoint_every = int(checkpoint_every) if checkpoint_every else None
        self.checkpoint_seconds = float(checkpoint_seconds) if checkpoint_seconds else None
        # Warnings JSON path
        self.domain_warnings_path = self.output_dir / 'domain_scheduler_warnings.json'

        # Progress logger (separate file; default to output logs dir)
        self.progress_logger = self.logger
        try:
            p = Path(progress_log_file) if progress_log_file else (self.logs_dir / 'download_progress.logs')
            p.parent.mkdir(parents=True, exist_ok=True)
            self.progress_logger = logging.getLogger(__name__ + ".progress")
            self.progress_logger.setLevel(progress_log_level)
            if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == str(p) for h in self.progress_logger.handlers):
                fh = logging.FileHandler(p)
                fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
                fh.setFormatter(fmt)
                fh.setLevel(progress_log_level)
                self.progress_logger.addHandler(fh)
            self.progress_logger.propagate = False
        except Exception:
            self.progress_logger = self.logger

        # Request-level logger to file (default to output logs dir)
        try:
            rp = Path(request_log_file) if request_log_file else (self.logs_dir / 'download_request.logs')
            rp.parent.mkdir(parents=True, exist_ok=True)
            if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == str(rp) for h in self.logger.handlers):
                rfh = logging.FileHandler(rp)
                rfh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                rfh.setLevel(self.logger.level)
                self.logger.addHandler(rfh)
        except Exception:
            pass

        # Ping/availability configuration
        self.pre_ping_domains = bool(pre_ping_domains)
        self.ping_timeout_s = float(ping_timeout_s)
        self.ping_concurrency = max(1, int(ping_concurrency))
        self.ping_method = ping_method.lower() if isinstance(ping_method, str) else 'head'
        self.ping_recheck_seconds = float(ping_recheck_seconds)
        self.down_wait_max_seconds = float(down_wait_max_seconds)
        # Parking/backoff knobs
        self.timeout_streak_threshold = int(timeout_streak_threshold)
        self.backoff_min_s = float(backoff_min_s)
        self.backoff_max_s = float(backoff_max_s)
        self.error_burst_window = int(error_burst_window)
        self.error_burst_threshold = float(error_burst_threshold)
        self.park_403_seconds = float(park_403_seconds)
        # TLS/SSL
        self.ssl_verify = bool(ssl_verify)
        self.ssl_cafile = ssl_cafile
        self.referer_column = str(referer_column) if referer_column else None
        # Per-domain SSL insecure override discovered via ping fallback
        self._domains_ssl_insecure: set[str] = set()

    def _mark_domain_ssl_insecure(self, domain: str) -> None:
        try:
            d = str(domain or '').strip()
            if not d:
                return
            if d not in self._domains_ssl_insecure:
                self._domains_ssl_insecure.add(d)
                self.progress_logger.info(f"[tls] Broken chain detected on {d}; enabling insecure SSL for this domain")
                # Persist into warnings JSON
                try:
                    import json
                    data = {'domains': []}
                    p = self.domain_warnings_path
                    if p.exists():
                        data = json.loads(p.read_text(encoding='utf-8') or '{}') or {'domains': []}
                    if 'domains' not in data or not isinstance(data['domains'], list):
                        data['domains'] = []
                    # Update or add entry
                    found = False
                    for rec in data['domains']:
                        if str(rec.get('base_domain') or '') == d:
                            rec['ssl_chain_broken'] = True
                            found = True
                            break
                    if not found:
                        data['domains'].append({'base_domain': d, 'ssl_chain_broken': True, 'note': 'SSL verify failed; insecure ping succeeded'})
                    p.parent.mkdir(parents=True, exist_ok=True)
                    tmp = p.with_suffix('.json.tmp')
                    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
                    os.replace(tmp, p)
                except Exception:
                    pass
        except Exception:
            pass
    
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
        used: Set[str] = set(self._used_filename_bases)
        used |= {
            b for b in df.get('filename_base', pd.Series([], dtype=str)).astype(str).tolist() if b
        }

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
    
    def get_base_url(self, url: str) -> str:
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

    def _extract_base_domain(self, url: str) -> str:
        """Synchronous helper: return scheme://netloc for a URL or empty string."""
        try:
            if not isinstance(url, str) or not url.strip():
                return ''
            u = url
            if not u.startswith(("http://", "https://")):
                u = f"https://{u}"
            p = urlparse(u)
            if not p.netloc:
                return ''
            scheme = p.scheme or 'https'
            return f"{scheme}://{p.netloc.lower()}"
        except Exception:
            return ''

    @dataclass
    class _DomainState:
        base: str
        queue: deque = field(default_factory=deque)
        active: int = 0
        concurrency: int = 1
        successes: int = 0
        failures: int = 0
        http_429: int = 0
        http_403: int = 0
        timeouts: int = 0
        durations: deque = field(default_factory=lambda: deque(maxlen=200))  # seconds per completed
        eta_exceeded_count: int = 0
        last_eta_seconds: float = 0.0
        warned: bool = False
        # Rolling error window
        recent_errors: deque = field(default_factory=lambda: deque(maxlen=50))  # strings like 'HTTP 429', 'Timeout'
        # Availability
        is_up: bool = True
        last_ping_ts: float = 0.0
        ping_failures: int = 0
        # Parking/backoff
        parked_until: Optional[float] = None
        timeout_streak: int = 0

        def avg_duration(self) -> float:
            if not self.durations:
                return 0.0
            return sum(self.durations) / len(self.durations)

        def remaining(self) -> int:
            # Pending in queue + currently active
            return len(self.queue) + self.active
    
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
        base_url = self.get_base_url(url)
        initial_url = base_url
        try:
            # Access the base domain to get cookies
            async with session.get(initial_url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)):
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
    
    async def make_request(self, session, requester, url, headers, timeout):
        """Make a request with tenacity retry logic and return content, status, headers"""
        from tenacity import AsyncRetrying
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max(1, int(self.max_retries))),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=(retry_if_exception_type(aiohttp.ClientError) |
                   retry_if_exception_type(asyncio.TimeoutError)),
            before_sleep=before_sleep_log(logging.getLogger(__name__), logging.INFO),
            reraise=True,
        ):
            with attempt:
                if requester == 'get':
                    async with session.get(url, headers=headers, timeout=timeout) as response:
                        response.raise_for_status()
                        content = await response.read()
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
    
    async def download_file(self, row_index: int, url: str, semaphore: Optional[asyncio.Semaphore], 
                           rate_limiter: RateLimiter, retry_count: int = 0,
                           filename_base: Optional[str] = None,
                           referer: Optional[str] = None) -> Tuple[bool, str, str, str, int]:
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
        base_url = self.get_base_url(url)
        
        # Enhanced headers with common browser-like attributes to bypass 403 errors
        # Prefer caller-provided referer (e.g., the external_link page)
        _referer = (referer or '').strip()
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            # Avoid brotli to prevent dependency errors
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'cross-site',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
            'TE': 'trailers',
            'Referer': _referer if _referer else f"https://www.google.com/search?q={domain}",
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
        
        if semaphore:
            await semaphore.acquire()
        try:
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
                # Prepare optional SSL connector
                connector = None
                # Domain-specific insecure override (discovered via ping)
                url_base = self._extract_base_domain(url)
                _force_insecure = url_base in getattr(self, '_domains_ssl_insecure', set())
                if (not self.ssl_verify) or _force_insecure:
                    connector = aiohttp.TCPConnector(ssl=False)
                elif self.ssl_cafile:
                    import ssl as _ssl
                    ctx = _ssl.create_default_context(cafile=self.ssl_cafile)
                    connector = aiohttp.TCPConnector(ssl=ctx)
                # Create a new session for each download to avoid cookie contamination
                async with aiohttp.ClientSession(cookies=cookies, connector=connector) as session:
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
                        
                        # Choose request method and perform streaming for GET
                        requester = self.request_method.lower()

                        try:
                            self.verbose_log(f"Attempting download request to URL: {url}")
                            self.verbose_log(f"Request method: {requester}, Timeout: {timeout.total}s")
                            self.verbose_log(f"Headers: {headers}")

                            if requester == 'get':
                                # Streaming GET with retries
                                from tenacity import AsyncRetrying
                                head = bytearray()
                                resp_headers = {}
                                async for attempt in AsyncRetrying(
                                    stop=stop_after_attempt(max(1, int(self.max_retries))),
                                    wait=wait_exponential(multiplier=1, min=1, max=10),
                                    retry=(retry_if_exception_type(aiohttp.ClientError) |
                                           retry_if_exception_type(asyncio.TimeoutError)),
                                    before_sleep=before_sleep_log(logging.getLogger(__name__), logging.INFO),
                                    reraise=True,
                                ):
                                    with attempt:
                                        async with session.get(url, headers=headers, timeout=timeout) as response:
                                            response.raise_for_status()
                                            resp_headers = dict(response.headers or {})
                                            # Write to a temp file first
                                            tmp_path = Path(self.downloads_dir) / f".part_{row_index}"
                                            async with aiofiles.open(tmp_path, 'wb') as f:
                                                async for chunk in response.content.iter_chunked(1 << 16):
                                                    if chunk:
                                                        if len(head) < (1 << 16):
                                                            need = (1 << 16) - len(head)
                                                            head.extend(chunk[:need])
                                                        await f.write(chunk)
                                            # Infer extension using URL, headers and first bytes
                                            file_ext = self.infer_file_extension(url, resp_headers, bytes(head))
                                            if not self.is_supported_format(file_ext):
                                                # Clean up temp and report
                                                try:
                                                    os.remove(tmp_path)
                                                except Exception:
                                                    pass
                                                self.logger.warning(f"Unsupported file format after inference: {file_ext}. Supported formats: {', '.join(self.supported_formats)}")
                                                return False, "", file_ext or "", f"Unsupported file format: {file_ext}", retry_count
                                            # Decide final filename
                                            if filename_base and str(filename_base).strip():
                                                filename = f"{filename_base}.{file_ext}"
                                            else:
                                                filename = self.generate_filename(row_index, file_ext)
                                            final_path = Path(self.downloads_dir) / filename
                                            try:
                                                os.replace(tmp_path, final_path)
                                            except Exception:
                                                # Fallback to copy/rename
                                                try:
                                                    os.rename(tmp_path, final_path)
                                                except Exception:
                                                    pass
                                            self.logger.info(f"Successfully downloaded {filename} from {url}")
                                            return True, filename, file_ext, "", retry_count
                            else:
                                # Fallback to non-streaming POST
                                content, status, resp_headers = await self.make_request(
                                    session, requester, url, headers, timeout
                                )
                                file_ext = self.infer_file_extension(url, resp_headers, content)
                                if not self.is_supported_format(file_ext):
                                    self.logger.warning(f"Unsupported file format after inference: {file_ext}. Supported formats: {', '.join(self.supported_formats)}")
                                    return False, "", file_ext or "", f"Unsupported file format: {file_ext}", retry_count
                                if filename_base and str(filename_base).strip():
                                    filename = f"{filename_base}.{file_ext}"
                                else:
                                    filename = self.generate_filename(row_index, file_ext)
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
                            
                            # Build error with optional Retry-After info
                            retry_after = None
                            try:
                                hdrs = dict(getattr(e, 'headers', {}) or {})
                                for k, v in hdrs.items():
                                    if k.lower() == 'retry-after':
                                        val = str(v).strip()
                                        if val.isdigit():
                                            retry_after = int(val)
                                        else:
                                            try:
                                                dt = parsedate_to_datetime(val)
                                                retry_after = max(0, int((dt.timestamp() - time.time())))
                                            except Exception:
                                                retry_after = None
                                        break
                            except Exception:
                                retry_after = None
                            error_msg = f"HTTP {status}: {str(e)}"
                            if status in (429, 503) and retry_after is not None:
                                error_msg += f" retry_after={retry_after}"
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
        finally:
            if semaphore:
                try:
                    semaphore.release()
                except Exception:
                    pass

    async def _download_files_async(self, df: pd.DataFrame, semaphore: asyncio.Semaphore, 
                                   rate_limiter: RateLimiter,
                                   *, checkpoint_path: Optional[Path] = None) -> pd.DataFrame:
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
        
        # Get total unprocessed rows (not downloaded successfully) and not duplicates
        mask = ~df['download_success']
        if 'is_duplicate' in df.columns:
            try:
                mask &= ~df['is_duplicate']
            except Exception:
                pass
        total_unprocessed = mask.sum()
        
        # Get indices for rows that need processing
        unprocessed_indices = df.index[mask].tolist()
        
        self.logger.info(f"Found {total_unprocessed} unprocessed rows out of {len(df)} total")
        
        # Variables to track progress
        successful_downloads = 0
        completed_since_ckpt = 0
        last_ckpt_ts = time.time()

        def _write_checkpoint() -> None:
            if not checkpoint_path:
                return
            try:
                df_out = df.copy()
                if '__url_norm' in df_out.columns:
                    try:
                        df_out = df_out.drop(columns=['__url_norm'])
                    except Exception:
                        pass
                tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + '.tmp')
                df_out.to_parquet(tmp, index=False)
                os.replace(tmp, checkpoint_path)
                self.logger.info(f"Checkpoint written: {checkpoint_path}")
            except Exception as e:
                self.logger.warning(f"Failed to write checkpoint {checkpoint_path}: {e}")
        
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
                # Optional per-row referer (e.g., external_link page)
                ref_val = None
                if self.referer_column and self.referer_column in df.columns:
                    try:
                        val = df.loc[row_idx, self.referer_column]
                        ref_val = str(val) if isinstance(val, str) and val.strip() else None
                    except Exception:
                        ref_val = None

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
                        filename_base=(df.loc[row_idx, 'filename_base'] if 'filename_base' in df.columns else None),
                        referer=ref_val
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
                    # Count completion for checkpoint cadence
                    completed_since_ckpt += 1

                    # Periodically save progress
                    if successful_downloads % self.save_every == 0 and successful_downloads > 0:
                        self.logger.info(f"Periodic save: Completed {successful_downloads} downloads.")
                    # Checkpoint by count or time
                    now = time.time()
                    if self.checkpoint_every and completed_since_ckpt >= self.checkpoint_every:
                        _write_checkpoint()
                        completed_since_ckpt = 0
                        last_ckpt_ts = now
                    elif self.checkpoint_seconds and (now - last_ckpt_ts) >= self.checkpoint_seconds:
                        _write_checkpoint()
                        completed_since_ckpt = 0
                        last_ckpt_ts = now
                    
                except Exception as e:
                    self.logger.error(f"Error processing task for row {row_idx}: {e}")
                    df.loc[row_idx, 'download_error'] = str(e)
                    df.loc[row_idx, 'download_retry_count'] += 1
                    completed_since_ckpt += 1
                    now = time.time()
                    if self.checkpoint_every and completed_since_ckpt >= self.checkpoint_every:
                        _write_checkpoint()
                        completed_since_ckpt = 0
                        last_ckpt_ts = now
                    elif self.checkpoint_seconds and (now - last_ckpt_ts) >= self.checkpoint_seconds:
                        _write_checkpoint()
                        completed_since_ckpt = 0
                        last_ckpt_ts = now
        # Final checkpoint
        try:
            _write_checkpoint()
        except Exception:
            pass
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

        # Compute base_domain column (scheme+host) for introspection and scheduling when needed
        if 'base_domain' not in df.columns:
            try:
                df['base_domain'] = df[self.url_column].map(self._extract_base_domain)
            except Exception:
                df['base_domain'] = ''

        # Ensure downloads directory exists
        os.makedirs(self.downloads_dir, exist_ok=True)
        
        # Use the already configured rate limiter if available
        rate_limiter = self.rate_limiter
        if not rate_limiter:
            rate_limiter = RateLimiter(100, 60, 30)  # Default limiter

        # Prepare checkpoint path (under output_dir/download_results)
        checkpoint_dir = self.output_dir / 'download_results'
        try:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        checkpoint_path = checkpoint_dir / f"download_results_{Path(input_parquet).name}.partial.parquet"

        async def _run_async():
            if self.scheduler_mode == 'per_domain':
                self.logger.info(
                    f"Starting per-domain scheduler: global_concurrency={self.concurrency}, "
                    f"per_domain_concurrency={self.per_domain_concurrency}"
                )
                return await self._download_files_async_per_domain(df, rate_limiter, checkpoint_path=checkpoint_path)
            else:
                semaphore = asyncio.Semaphore(self.concurrency)
                self.logger.info(
                    f"Starting download (global) with concurrency={self.concurrency}, "
                    f"rate_limit={rate_limiter.rate_limit}/{rate_limiter.time_period}s"
                )
                return await self._download_files_async(df, semaphore, rate_limiter, checkpoint_path=checkpoint_path)

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running and running.is_running():
            raise RuntimeError("An event loop is already running; use the async variant `adownload_files`.")
        updated_df = asyncio.run(_run_async())
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

    async def _download_files_async_per_domain(self, df: pd.DataFrame, rate_limiter: RateLimiter, *, checkpoint_path: Optional[Path] = None) -> pd.DataFrame:
        """Per-domain round-robin scheduler with dynamic concurrency and ETA tracking."""
        df = df.copy()
        completed_since_ckpt = 0
        last_ckpt_ts = time.time()

        def _write_checkpoint() -> None:
            if not checkpoint_path:
                return
            try:
                df_out = df.copy()
                if '__url_norm' in df_out.columns:
                    try:
                        df_out = df_out.drop(columns=['__url_norm'])
                    except Exception:
                        pass
                tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + '.tmp')
                df_out.to_parquet(tmp, index=False)
                os.replace(tmp, checkpoint_path)
                self.logger.info(f"Checkpoint written: {checkpoint_path}")
            except Exception as e:
                self.logger.warning(f"Failed to write checkpoint {checkpoint_path}: {e}")

        # Initialize result columns if they don't exist
        if 'download_success' not in df.columns:
            df['download_success'] = False
        if 'filename' not in df.columns:
            df['filename'] = ""
        if 'download_error' not in df.columns:
            df['download_error'] = ""
        if 'download_retry_count' not in df.columns:
            df['download_retry_count'] = 0
        if 'file_ext' not in df.columns:
            df['file_ext'] = ""
        if 'is_duplicate' not in df.columns:
            df['is_duplicate'] = False
        if 'duplicate_of' not in df.columns:
            df['duplicate_of'] = ""
        if 'source_row' not in df.columns:
            try:
                df['source_row'] = df.index.astype('int64')
            except Exception:
                df['source_row'] = 0
        if 'url_index' not in df.columns:
            df['url_index'] = 0

        # Build domain queues for unprocessed, non-duplicate rows
        mask = (~df['download_success']) & (~df['is_duplicate'])
        row_indices = df.index[mask].tolist()
        domains: Dict[str, GlossDownloader._DomainState] = {}
        for idx in row_indices:
            url = df.at[idx, self.url_column]
            # Determine grouping key
            if self.scheduler_group_by and self.scheduler_group_by != 'base_domain':
                key = str(df.at[idx, self.scheduler_group_by]) if self.scheduler_group_by in df.columns else ''
            else:
                key = df.at[idx, 'base_domain'] if 'base_domain' in df.columns else self._extract_base_domain(url)
            if not isinstance(key, str):
                key = str(key) if key is not None else ''
            if not key:
                key = ''
            if key not in domains:
                # Each group starts with up to per_domain_concurrency, but not exceeding global
                start_c = min(self.per_domain_concurrency, max(1, self.concurrency))
                domains[key] = GlossDownloader._DomainState(base=key, concurrency=start_c)
            domains[key].queue.append(idx)

        if not domains:
            self.logger.info("No rows to process (per-domain)")
            return df

        uniq_domains = list(domains.keys())

        # Pre-ping domains to check availability (only meaningful if grouping by base_domain)
        async def _ping_one(session: aiohttp.ClientSession, base: str) -> bool:
            if not base:
                return False
            try:
                timeout = aiohttp.ClientTimeout(total=self.ping_timeout_s)
                headers = {'User-Agent': next(self.user_agents)}
                if self.ping_method == 'get':
                    async with session.get(base, headers=headers, timeout=timeout, allow_redirects=True) as r:
                        # consider reachable for any HTTP status; connection success is the key
                        return True
                else:
                    # HEAD first, fallback to GET on 405
                    async with session.head(base, headers=headers, timeout=timeout, allow_redirects=True) as r:
                        return True
            except aiohttp.ClientResponseError as e:
                if getattr(e, 'status', 0) == 405:
                    try:
                        async with session.get(base, headers={'User-Agent': next(self.user_agents)}, timeout=aiohttp.ClientTimeout(total=self.ping_timeout_s), allow_redirects=True) as r:
                            return True
                    except Exception:
                        return False
                return False
            except Exception:
                return False

        async def ping_all(dom_list: List[str]) -> Dict[str, bool]:
            results: Dict[str, bool] = {}
            sem = asyncio.Semaphore(self.ping_concurrency)
            async with aiohttp.ClientSession() as session:
                async def _task(d: str):
                    async with sem:
                        ok = await _ping_one(session, d)
                        # If strict ping failed, try insecure ping once to detect broken chain
                        if not ok and d:
                            try:
                                import aiohttp as _aio
                                conn = _aio.TCPConnector(ssl=False)
                                to = _aio.ClientTimeout(total=self.ping_timeout_s)
                                async with _aio.ClientSession(connector=conn) as s2:
                                    try:
                                        # HEAD then GET fallback
                                        try:
                                            async with s2.head(d, headers={'User-Agent': next(self.user_agents)}, timeout=to, allow_redirects=True):
                                                ok2 = True
                                        except Exception:
                                            async with s2.get(d, headers={'User-Agent': next(self.user_agents)}, timeout=to, allow_redirects=True):
                                                ok2 = True
                                    except Exception:
                                        ok2 = False
                                if ok2:
                                    # mark domain as ssl-broken and usable in insecure mode
                                    self._mark_domain_ssl_insecure(d)
                                    ok = True
                            except Exception:
                                pass
                        results[d] = ok
                await asyncio.gather(*[_task(d) for d in dom_list])
            return results

        # Perform initial ping if enabled
        ping_results: Dict[str, bool] = {}
        if self.pre_ping_domains and uniq_domains and self.scheduler_group_by == 'base_domain':
            try:
                self.progress_logger.info(f"[ping] Pinging {len(uniq_domains)} base domains before scheduling …")
                ping_results = await ping_all(uniq_domains)
                up = sum(1 for v in ping_results.values() if v)
                down = len(ping_results) - up
                self.progress_logger.info(f"[ping] Reachable domains: {up}, Unreachable: {down}")
            except Exception as e:
                self.progress_logger.warning(f"[ping] Initial domain ping failed: {e}")

        # Mark domain availability
        now_ts = time.time()
        for d, st in domains.items():
            if ping_results:
                ok = bool(ping_results.get(d, True))
                st.is_up = ok
                st.last_ping_ts = now_ts
            else:
                st.is_up = True
                st.last_ping_ts = now_ts

        # Auto-compute max_active_domains if not provided: keep within global cap
        if self.max_active_domains is None:
            # How many domains can be active if each may take up to per_domain_concurrency
            denom = max(1, self.per_domain_concurrency)
            mad = max(1, self.concurrency // denom)
            self.max_active_domains = max(1, min(mad, len(uniq_domains)))

        # Active/pending domain rings (prefer 'up' domains first)
        up_domains = [d for d in uniq_domains if domains[d].is_up]
        down_domains = [d for d in uniq_domains if not domains[d].is_up]
        active_order = deque(up_domains[: self.max_active_domains])
        pending_domains = deque(up_domains[self.max_active_domains :])
        # Keep down ones at the tail; we'll recheck periodically
        pending_down = deque(down_domains)

        # Global in-flight limit
        global_in_flight = 0
        max_global = max(1, self.concurrency)

        # Track running tasks
        tasks: Dict[asyncio.Task, Tuple[str, int, float]] = {}

        # For warnings
        warnings: Dict[str, Dict[str, Any]] = {}

        # Progress tracking
        total_rows = sum(len(state.queue) for state in domains.values())
        global_completed = 0
        last_log_t = time.time()
        log_interval_s = 10.0

        def snapshot_progress() -> Dict[str, Any]:
            doms = []
            for d, st in domains.items():
                doms.append({
                    'base_domain': d,
                    'remaining': st.remaining(),
                    'active': st.active,
                    'concurrency': st.concurrency,
                    'successes': st.successes,
                    'failures': st.failures,
                    'avg_duration_s': round(st.avg_duration(), 3),
                    'eta_seconds': round(estimate_eta_s(st), 3),
                    'is_up': st.is_up,
                    'parked_until': st.parked_until or 0.0,
                })
            doms.sort(key=lambda x: (x['eta_seconds'], -x['remaining']), reverse=True)
            return {
                'timestamp': time.time(),
                'total_rows': total_rows,
                'completed': global_completed,
                'in_flight': global_in_flight,
                'active_domains': len([d for d in active_order if d in domains]),
                'pending_domains': len(pending_domains),
                'pending_down_domains': len(pending_down),
                'domains': doms,
            }

        def _fmt_hms(seconds: float) -> str:
            try:
                s = int(max(0, seconds))
                h, r = divmod(s, 3600)
                m, s = divmod(r, 60)
                if h:
                    return f"{h:02d}:{m:02d}:{s:02d}"
                return f"{m:02d}:{s:02d}"
            except Exception:
                return "00:00"

        def maybe_log_and_write_progress(force: bool = False):
            nonlocal last_log_t
            now = time.time()
            if not force and (now - last_log_t) < log_interval_s:
                return
            prog = snapshot_progress()
            # Write progress JSON
            try:
                p = self.output_dir / 'domain_progress.json'
                tmp = p.with_suffix('.json.tmp')
                with open(tmp, 'w', encoding='utf-8') as f:
                    json.dump(prog, f, ensure_ascii=False, indent=2)
                os.replace(tmp, p)
            except Exception:
                pass
            # High-level logging with percentages, progress bars, ETAs
            total = int(prog.get('total_rows') or 0)
            completed = int(prog.get('completed') or 0)
            inflight = int(prog.get('in_flight') or 0)
            # Aggregate success/fail across domains
            succ = sum(int(d.get('successes', 0)) for d in prog.get('domains', []))
            fail = sum(int(d.get('failures', 0)) for d in prog.get('domains', []))
            rem = sum(int(d.get('remaining', 0)) for d in prog.get('domains', []))
            # Global ETA approximations
            # Sum of work durations divided by global concurrency
            sum_work = 0.0
            max_eta = 0.0
            for d in prog.get('domains', []):
                rem_i = int(d.get('remaining', 0))
                avg_i = float(d.get('avg_duration_s') or 0.0)
                sum_work += rem_i * max(0.0, avg_i)
                max_eta = max(max_eta, float(d.get('eta_seconds') or 0.0))
            est_total_eta = sum_work / max(1, max_global)
            # Header line
            succ_pct = (succ / max(1, succ + fail)) * 100.0
            fail_pct = (fail / max(1, succ + fail)) * 100.0
            # Domain status buckets for clarity
            domains_data = list(prog.get('domains', []))
            running = [d for d in domains_data if int(d.get('active', 0)) > 0]
            waiting_up = [d for d in domains_data if int(d.get('active', 0)) == 0 and int(d.get('remaining', 0)) > 0 and bool(d.get('is_up', True))]
            waiting_down = [d for d in domains_data if int(d.get('active', 0)) == 0 and int(d.get('remaining', 0)) > 0 and not bool(d.get('is_up', True))]
            done_domains = [d for d in domains_data if int(d.get('remaining', 0)) == 0 and int(d.get('active', 0)) == 0]
            # Snapshot delimiter with counts
            domains_left = len(running) + len(waiting_up) + len(waiting_down)
            self.progress_logger.info(
                f"===== Snapshot: {completed}/{total} tasks (in_flight={inflight}, active_domains={prog.get('active_domains')}) | "
                f"domains: running={len(running)} waiting_up={len(waiting_up)} waiting_down={len(waiting_down)} done={len(done_domains)} left={domains_left} | "
                f"succ={succ} ({succ_pct:.1f}%) fail={fail} ({fail_pct:.1f}%) rem={rem} | total_eta≈{_fmt_hms(est_total_eta)} critical_eta≈{_fmt_hms(max_eta)}"
            )

            # Helper to render one domain line with status tag
            def render_domain(d: Dict[str, Any], status: str) -> str:
                name = (d.get('base_domain') or '<nohost>')
                total_i = int(d.get('successes', 0)) + int(d.get('failures', 0)) + int(d.get('remaining', 0))
                done_i = int(d.get('successes', 0))
                fail_i = int(d.get('failures', 0))
                rem_i = int(d.get('remaining', 0))
                act_i = int(d.get('active', 0))
                conc_i = int(d.get('concurrency', 0))
                eta_i = float(d.get('eta_seconds') or 0.0)
                avg_i = float(d.get('avg_duration_s') or 0.0)
                p_done = (done_i / max(1, total_i))
                p_fail = (fail_i / max(1, total_i))
                bar_len = 24
                done_chars = int(p_done * bar_len)
                fail_chars = int(p_fail * bar_len)
                rem_chars = max(0, bar_len - done_chars - fail_chars)
                bar = '[' + ('=' * done_chars) + ('!' * fail_chars) + ('.' * rem_chars) + ']'
                return (
                    f"[{status}] {name} {bar} {done_i}/{total_i} | fail%={p_fail*100:4.1f} | act={act_i}/{conc_i} | eta={_fmt_hms(eta_i)} | avg={avg_i:.2f}s"
                )

            # Print groups: running first, then waiting (up), then a small sample of down and done
            if running:
                self.progress_logger.info("-- Running domains (%d) --", len(running))
                for d in sorted(running, key=lambda x: int(x.get('remaining', 0)), reverse=True):
                    self.progress_logger.info(render_domain(d, 'RUN'))
            if waiting_up:
                self.progress_logger.info("-- Waiting (up) domains (%d) --", len(waiting_up))
                for d in sorted(waiting_up, key=lambda x: int(x.get('remaining', 0)), reverse=True)[:20]:
                    self.progress_logger.info(render_domain(d, 'WAIT'))
            if waiting_down:
                self.progress_logger.info("-- Waiting (down) domains (%d) --", len(waiting_down))
                for d in sorted(waiting_down, key=lambda x: int(x.get('remaining', 0)), reverse=True)[:10]:
                    self.progress_logger.info(render_domain(d, 'DOWN'))
            # Keep done section compact to avoid noise
            if done_domains:
                self.progress_logger.info("-- Done domains (%d) --", len(done_domains))
                # show a small tail of done list
                for d in sorted(done_domains, key=lambda x: int(x.get('successes', 0)) + int(x.get('failures', 0)))[:5]:
                    self.progress_logger.info(render_domain(d, 'DONE'))
            last_log_t = now

        def estimate_eta_s(state: GlossDownloader._DomainState) -> float:
            remaining = state.remaining()
            if remaining <= 0:
                return 0.0
            avg = state.avg_duration() or 5.0  # default initial guess
            eff_c = max(self.domain_concurrency_floor, min(state.concurrency, self.domain_concurrency_ceiling))
            # ETA ≈ remaining * avg / eff_c (assuming steady parallelism)
            return float(remaining) * avg / max(1, eff_c)

        def should_ease(state: GlossDownloader._DomainState) -> bool:
            # Consider easing if frequent timeouts/HTTP 429/403 or high avg latency
            if not self.dynamic_tuning:
                return False
            recent = list(state.recent_errors)
            if not recent:
                return False
            n = len(recent)
            timeouts = sum(1 for e in recent if 'Timeout' in e)
            ratelims = sum(1 for e in recent if 'HTTP 429' in e or 'HTTP 503' in e)
            forbids = sum(1 for e in recent if 'HTTP 403' in e)
            # Thresholds: 20% timeouts or 20% 429/503, or avg > 60s
            if timeouts / n >= 0.2:
                return True
            if (ratelims + forbids) / n >= 0.2:
                return True
            if (state.avg_duration() or 0) > 60:
                return True
            return False

        async def dispatch_ready():
            nonlocal global_in_flight
            # Fill capacity while we can
            made_progress = False
            if not active_order:
                return False
            # Iterate up to len(active_order) to attempt a full RR rotation
            for _ in range(len(active_order)):
                if global_in_flight >= max_global:
                    break
                dom = active_order[0]
                state = domains.get(dom)
                if state is None:
                    active_order.popleft()
                    continue
                # Skip if domain is down or parked
                now = time.time()
                if not state.is_up:
                    # Move out of active to free slot and try to promote a pending
                    try:
                        active_order.popleft()
                    except Exception:
                        pass
                    if dom not in pending_down:
                        pending_down.append(dom)
                    if pending_domains:
                        active_order.append(pending_domains.popleft())
                    # Continue to next active candidate
                    continue
                if state.parked_until is not None and now < state.parked_until:
                    # Temporarily move to pending to free the slot
                    try:
                        active_order.popleft()
                    except Exception:
                        pass
                    if dom not in pending_domains:
                        pending_domains.append(dom)
                    if pending_domains:
                        # bring next pending up
                        active_order.append(pending_domains.popleft())
                    continue
                if state.parked_until is not None and now >= state.parked_until:
                    # Unpark with base ping
                    state.parked_until = None
                    try:
                        pres = await ping_all([dom])
                        ok = bool(pres.get(dom, False))
                    except Exception:
                        ok = False
                    if not ok:
                        state.is_up = False
                        if dom not in pending_down:
                            pending_down.append(dom)
                        try:
                            active_order.popleft()
                        except Exception:
                            pass
                        if pending_domains:
                            active_order.append(pending_domains.popleft())
                        continue
                    state.concurrency = max(self.domain_concurrency_floor, 1)
                    self.progress_logger.info(f"[park] Unparked domain: {dom}; resuming at concurrency={state.concurrency}")
                # Attempt to launch up to (state.concurrency - state.active)
                while (
                    global_in_flight < max_global and
                    state.active < state.concurrency and
                    state.queue
                ):
                    row_idx = state.queue.popleft()
                    url = df.at[row_idx, self.url_column]
                    retry_count = int(df.at[row_idx, 'download_retry_count']) if 'download_retry_count' in df.columns else 0
                    # Skip rows with too many failures
                    if retry_count >= self.skip_failed_after:
                        continue
                    # Launch task
                    t0 = time.time()
                    # Optional per-row referer
                    ref_val = None
                    if self.referer_column and self.referer_column in df.columns:
                        try:
                            val = df.at[row_idx, self.referer_column]
                            ref_val = str(val) if isinstance(val, str) and val.strip() else None
                        except Exception:
                            ref_val = None
                    task = asyncio.create_task(self.download_file(
                        row_index=row_idx,
                        url=url,
                        semaphore=None,
                        rate_limiter=rate_limiter,
                        retry_count=retry_count,
                        filename_base=(df.at[row_idx, 'filename_base'] if 'filename_base' in df.columns else None),
                        referer=ref_val
                    ))
                    tasks[task] = (dom, row_idx, t0)
                    state.active += 1
                    global_in_flight += 1
                    made_progress = True
                    if global_in_flight >= max_global:
                        break
                # RR rotate
                active_order.rotate(-1)
            return made_progress

        # Initial log explaining single-thread supports 5 concurrency
        if self.max_active_domains == 1 and self.per_domain_concurrency >= 5:
            self.logger.info(
                "Per-domain mode: single active domain will use up to 5 concurrent requests via asyncio."
            )

        # Main loop
        while tasks or any(domains[d].queue or domains[d].active for d in list(active_order) + list(pending_domains)):
            # Try to dispatch initially
            await dispatch_ready()

            if not tasks:
                # If there are no tasks but queues remain, ensure we promote domains
                # Remove drained domains from active and promote pending
                drained = [d for d in list(active_order) if not domains[d].queue and domains[d].active == 0]
                for d in drained:
                    try:
                        active_order.remove(d)
                    except Exception:
                        pass
                    if pending_domains:
                        active_order.append(pending_domains.popleft())
                # Try dispatch again
                await dispatch_ready()
                # If nothing to do and only down domains remain, wait up to configured window
                if not tasks and not any(domains[d].queue for d in domains if d in active_order or d in pending_domains):
                    if pending_down:
                        if 'down_wait_started_ts' not in locals() or down_wait_started_ts is None:
                            down_wait_started_ts = time.time()
                            self.progress_logger.info(
                                f"[ping] All up domains are drained; waiting up to {int(self.down_wait_max_seconds)}s for any down domain to recover …"
                            )
                        else:
                            waited = time.time() - down_wait_started_ts
                            if waited >= self.down_wait_max_seconds:
                                self.progress_logger.warning(
                                    f"[ping] No domain recovered after {int(waited)}s; terminating run with {len(pending_down)} domains still down."
                                )
                                break
                        # Proactively recheck a batch now
                        try:
                            sample = list(list(pending_down)[:min(10, len(pending_down))])
                            res = await _ping_all_aligned(sample)
                            for d, ok in res.items():
                                if ok:
                                    try:
                                        pending_down.remove(d)
                                    except Exception:
                                        pass
                                    if len(active_order) < self.max_active_domains:
                                        active_order.append(d)
                                    else:
                                        pending_domains.append(d)
                                    domains[d].is_up = True
                                    domains[d].last_ping_ts = time.time()
                                    self.progress_logger.info(
                                        f"[ping] Domain recovered: {d}; pending={len(pending_domains)} down={len(pending_down)}"
                                    )
                        except Exception:
                            pass
                        await asyncio.sleep(0.5)
                        continue
                    else:
                        # Nothing left anywhere
                        break

            if not tasks:
                # Nothing to wait on
                await asyncio.sleep(0.05)
                continue

            # Optionally recheck down domains periodically
            # Track ping cadence independently from logging cadence
            if 'last_ping_recheck_ts' not in locals():
                last_ping_recheck_ts = 0.0
            if self.pre_ping_domains and pending_down and (time.time() - last_ping_recheck_ts) >= self.ping_recheck_seconds:
                try:
                    # Check a slice to avoid bursts
                    sample = list(list(pending_down)[:min(10, len(pending_down))])
                    res = await _ping_all_aligned(sample)
                    for d, ok in res.items():
                        if ok:
                            # promote to pending_domains
                            try:
                                pending_down.remove(d)
                            except Exception:
                                pass
                            if d not in pending_domains and d not in active_order:
                                # If there's room in active_order, place it directly
                                if len(active_order) < self.max_active_domains:
                                    active_order.append(d)
                                else:
                                    pending_domains.append(d)
                                domains[d].is_up = True
                                domains[d].last_ping_ts = time.time()
                                self.progress_logger.info(f"[ping] Domain recovered: {d}; pending={len(pending_domains)} down={len(pending_down)}")
                        else:
                            domains[d].is_up = False
                            domains[d].last_ping_ts = time.time()
                except Exception:
                    pass
                last_ping_recheck_ts = time.time()

            # Wait for at least one to complete
            done, _ = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)

            # Process completions
            for task in done:
                dom, row_idx, t0 = tasks.pop(task)
                dt = max(0.0, time.time() - t0)
                state = domains.get(dom)
                if state is None:
                    continue
                state.active = max(0, state.active - 1)
                global_in_flight = max(0, global_in_flight - 1)

                try:
                    success, filename, file_ext, error, retry_count = await task
                except Exception as e:
                    success, filename, file_ext, error, retry_count = False, "", "", str(e), int(df.at[row_idx, 'download_retry_count']) + 1

                # Update DataFrame
                df.at[row_idx, 'download_success'] = bool(success)
                df.at[row_idx, 'filename'] = filename
                df.at[row_idx, 'file_ext'] = file_ext
                df.at[row_idx, 'download_error'] = error
                df.at[row_idx, 'download_retry_count'] = retry_count

                # Update metrics
                state.durations.append(dt)
                if success:
                    state.successes += 1
                    state.timeout_streak = 0
                else:
                    state.failures += 1
                    e = str(error or '')
                    state.recent_errors.append(e)
                    if 'HTTP 429' in e or 'HTTP 503' in e:
                        state.http_429 += 1
                    if 'HTTP 403' in e:
                        state.http_403 += 1
                    if 'Timeout' in e:
                        state.timeouts += 1
                        state.timeout_streak = state.timeout_streak + 1
                    else:
                        state.timeout_streak = 0
                global_completed += 1
                completed_since_ckpt += 1

                # Dynamic tuning: ease if overloaded
                if self.dynamic_tuning and should_ease(state):
                    if state.concurrency > self.domain_concurrency_floor:
                        state.concurrency -= 1
                        self.logger.info(f"Easing concurrency for {dom} -> {state.concurrency}")

                # Parking/backoff based on errors
                if not success:
                    now2 = time.time()
                    # 429/503 -> respect Retry-After if present in error
                    if ('HTTP 429' in e or 'HTTP 503' in e):
                        retry_after = None
                        try:
                            import re
                            m = re.search(r"retry_after=(\d+)", e)
                            if m:
                                retry_after = int(m.group(1))
                        except Exception:
                            retry_after = None
                        if retry_after is None:
                            retry_after = max(1, int(self.ping_recheck_seconds))
                        state.parked_until = now2 + retry_after
                        state.concurrency = max(self.domain_concurrency_floor, 1)
                        self.progress_logger.info(f"[park] Rate limited: {dom}; parked for {retry_after}s")
                    # Timeout streak -> exponential backoff
                    elif state.timeout_streak >= int(getattr(self, 'timeout_streak_threshold', 5)):
                        backoff = min(float(getattr(self, 'backoff_min_s', 60.0)) * (2 ** max(0, state.ping_failures)), float(getattr(self, 'backoff_max_s', 900.0)))
                        state.ping_failures += 1
                        state.parked_until = now2 + backoff
                        state.concurrency = max(self.domain_concurrency_floor, 1)
                        state.timeout_streak = 0
                        self.progress_logger.info(f"[park] Timeout streak: {dom}; parked for {int(backoff)}s (level={state.ping_failures})")
                    else:
                        # 403 burst heuristic -> short park
                        window = list(state.recent_errors)
                        if window:
                            c403 = sum(1 for x in window if 'HTTP 403' in x)
                            frac403 = c403 / len(window)
                            if frac403 >= float(getattr(self, 'error_burst_threshold', 0.5)) and len(window) >= int(getattr(self, 'error_burst_window', 20)):
                                park_s = float(getattr(self, 'park_403_seconds', 600.0))
                                state.parked_until = now2 + park_s
                                state.concurrency = 1
                                self.progress_logger.info(f"[park] 403 burst: {dom}; parked for {int(park_s)}s")

                # ETA handling and promotion/rotation
                eta_s = estimate_eta_s(state)
                state.last_eta_seconds = eta_s
                if eta_s >= self.eta_max_seconds:
                    state.eta_exceeded_count += 1
                    if state.eta_exceeded_count == 1:
                        # Try to increase concurrency gently to improve ETA, up to ceiling
                        if state.concurrency < self.domain_concurrency_ceiling:
                            state.concurrency += 1
                            self.logger.info(
                                f"ETA high for {dom} ({int(eta_s)}s). Bumping concurrency -> {state.concurrency}"
                            )
                    elif state.eta_exceeded_count >= 2 and not state.warned:
                        # Persist warning
                        warnings[dom] = {
                            'base_domain': dom,
                            'eta_seconds': eta_s,
                            'avg_duration_seconds': state.avg_duration(),
                            'queue_remaining': len(state.queue),
                            'active': state.active,
                            'concurrency': state.concurrency,
                            'successes': state.successes,
                            'failures': state.failures,
                            'http_429': state.http_429,
                            'http_403': state.http_403,
                            'timeouts': state.timeouts,
                            'note': 'ETA threshold exceeded twice; domain may be problematic.'
                        }
                        state.warned = True
                        try:
                            with open(self.domain_warnings_path, 'w', encoding='utf-8') as f:
                                json.dump({'domains': list(warnings.values())}, f, ensure_ascii=False, indent=2)
                            self.logger.warning(
                                f"Wrote domain warnings to {self.domain_warnings_path}"
                            )
                        except Exception as we:
                            self.logger.error(f"Failed to write warnings JSON: {we}")

                # If domain is drained, rotate it out and promote a pending domain
                if state.active == 0 and not state.queue:
                    # Remove if present
                    if dom in active_order:
                        try:
                            active_order.remove(dom)
                        except Exception:
                            pass
                    # Prefer 'up' pending domains
                    next_dom = None
                    while pending_domains and next_dom is None:
                        cand = pending_domains.popleft()
                        if domains.get(cand) and domains[cand].is_up:
                            next_dom = cand
                            break
                    if next_dom:
                        active_order.append(next_dom)
                        try:
                            self.progress_logger.info(
                                f"[rr] Domain drained: {dom or '<nohost>'}; activated: {next_dom or '<nohost>'}; "
                                f"active_domains={len(active_order)} pending={len(pending_domains)} down={len(pending_down)}"
                            )
                        except Exception:
                            pass

            # Periodic checkpoint by cadence
            try:
                now_ck = time.time()
                if self.checkpoint_every and completed_since_ckpt >= self.checkpoint_every:
                    _write_checkpoint()
                    completed_since_ckpt = 0
                    last_ckpt_ts = now_ck
                elif self.checkpoint_seconds and (now_ck - last_ckpt_ts) >= self.checkpoint_seconds:
                    _write_checkpoint()
                    completed_since_ckpt = 0
                    last_ckpt_ts = now_ck
            except Exception:
                pass

            # After processing completions, try dispatch again
            await dispatch_ready()
            # Periodic progress log
            maybe_log_and_write_progress()

        # Final checkpoint and progress snapshot
        try:
            _write_checkpoint()
            maybe_log_and_write_progress(force=True)
        except Exception:
            pass
        return df

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

        col_vals = df[self.url_column]
        def _looks_expand(v: Any) -> bool:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return False
            if isinstance(v, (list, tuple, dict)):
                return True
            if isinstance(v, str):
                s = v.strip()
                return s.startswith('[') or s.startswith('{')
            return False
        try:
            needs_expand = bool(col_vals.apply(_looks_expand).any())
        except Exception:
            needs_expand = False

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

    async def adownload_files(self, input_parquet: str, **kwargs) -> pd.DataFrame:
        """Async variant of download_files for notebook/async contexts."""
        self.logger.info(f"Loading parquet file: {input_parquet}")
        df = pd.read_parquet(input_parquet)
        if self.url_column not in df.columns:
            available_columns = df.columns.tolist()
            self.logger.error(f"URL column '{self.url_column}' not found in parquet file. Available columns: {available_columns}")
            url_like_columns = [col for col in available_columns if any(keyword in col.lower() for keyword in ['url', 'link', 'uri'])]
            if url_like_columns:
                self.url_column = url_like_columns[0]
                self.logger.warning(f"Using '{self.url_column}' as URL column instead")
            else:
                raise ValueError(f"URL column '{self.url_column}' not found in parquet file and no alternative URL columns detected")
        df = self._expand_and_mark_duplicates(df)
        df = self.ensure_filename_base(df)
        if 'base_domain' not in df.columns:
            try:
                df['base_domain'] = df[self.url_column].map(self._extract_base_domain)
            except Exception:
                df['base_domain'] = ''
        os.makedirs(self.downloads_dir, exist_ok=True)
        rate_limiter = self.rate_limiter or RateLimiter(100, 60, 30)
        # Prepare checkpoint path
        checkpoint_dir = self.output_dir / 'download_results'
        try:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        checkpoint_path = checkpoint_dir / f"download_results_{Path(input_parquet).name}.partial.parquet"
        if self.scheduler_mode == 'per_domain':
            self.logger.info(
                f"Starting per-domain scheduler: global_concurrency={self.concurrency}, per_domain_concurrency={self.per_domain_concurrency}"
            )
            updated_df = await self._download_files_async_per_domain(df, rate_limiter, checkpoint_path=checkpoint_path)
        else:
            semaphore = asyncio.Semaphore(self.concurrency)
            self.logger.info(
                f"Starting download (global) with concurrency={self.concurrency}, rate_limit={rate_limiter.rate_limit}/{rate_limiter.time_period}s"
            )
            updated_df = await self._download_files_async(df, semaphore, rate_limiter, checkpoint_path=checkpoint_path)
        if '__url_norm' in updated_df.columns:
            try:
                updated_df = updated_df.drop(columns=['__url_norm'])
            except Exception:
                pass
        success_count = updated_df['download_success'].sum()
        fail_count = len(updated_df) - success_count
        self.logger.info(f"Download complete: {success_count} successful, {fail_count} failed, files downloaded to {self.downloads_dir}")
        return updated_df
