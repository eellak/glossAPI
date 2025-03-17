import aiohttp
import asyncio
import os
import argparse
from urllib.parse import urlparse
import random
import aiofiles
import logging
import json
import time 

#Configure logging for behavior tracking and errors 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_indexes(papers):
    """
    Get the highest index of papers downloaded for continuation.
    
    Args:
        papers: List of paper filenames or keys
        
    Returns:
        List containing the highest index found, or empty list if no files
    """
    if papers:
        nums = []
        for p in papers:
            num = p.split("_")[-1]
            nums.append(int(num))
        return sorted(nums)[-1:]
    return []

async def fetch_all_pages(base_url, params, headers, semaphore):
    """
    Fetch all paginated results from an API endpoint.
    
    Args:
        base_url: The API URL to fetch from
        params: Dictionary of query parameters
        headers: Request headers
        semaphore: Semaphore to limit concurrent requests
        
    Returns:
        Combined metadata dictionary from all pages
    """
    all_metadata = {}
    page = 1
    has_more_data = True
    
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        while has_more_data:
            params['page'] = page
            
            async with semaphore:
                try:
                    logging.info(f"Fetching page {page} from {base_url}")
                    async with session.get(base_url, params=params, headers=headers) as response:
                        if response.status != 200:
                            logging.error(f"Failed to fetch page {page}: Status code {response.status}")
                            break
                            
                        page_data = await response.json()
                        
                        if not page_data:
                            logging.info(f"No more data found after page {page-1}")
                            has_more_data = False
                            break
                            
                        # Update metadata with current page's data
                        all_metadata.update(page_data)
                        logging.info(f"Fetched {len(page_data)} items from page {page}")
                        page += 1
                        
                        # Small pause between requests to avoid rate limiting
                        await asyncio.sleep(0.5)
                        
                except aiohttp.ClientError as e:
                    logging.error(f"Client error fetching page {page}: {e}")
                    break
                except Exception as e:
                    logging.error(f"Unexpected error fetching page {page}: {e}")
                    break
    
    logging.info(f"Total metadata entries fetched: {len(all_metadata)}")
    return all_metadata

async def download_pdfs(metadata_dict, semaphore, visited, indexes, args, progress_report, retry=3):
    """
    Prepares tasks for download_pdf function and stores association of "paper_name.pdf" with original metadata.
    
    Args:
        metadata_dict: Dictionary containing metadata and URLs
        semaphore: Semaphore to limit concurrent downloads
        visited: List of already downloaded files
        indexes: List of indexes for naming files sequentially
        args: Command-line arguments
        progress_report: Dictionary to track download progress
        retry: Number of retry attempts
        
    Returns:
        Boolean indicating if all files have been processed
    """
    retry -= 1
    retries = {} #Dictionary holding files for download retrial
    tasks = [] #List to hold the tasks to be executed
    ordered_metadata = list(metadata_dict.items())
    user_agent_gen = user_agent_generator()
    i = 0
    reached_end_of_file = True #flag: if all metadata are in "visited"
    
    #Process metadata urls and schedule downloads
    for metadata, url in ordered_metadata:
        if i < args.batch and metadata not in visited:
            reached_end_of_file = False
            if indexes:
                index = indexes[-1] + 1
            else:
                index = 1
            indexes.append(index)
            task = asyncio.create_task(
                download_pdf(index, metadata, url, semaphore, args, next(user_agent_gen))
            )
            tasks.append(task)
            i += 1
    results = await asyncio.gather(*tasks)
    for r in results:
        if r:
            has_downloaded_file, metadata, pdf_file_name = r
            if has_downloaded_file:
                progress_report[pdf_file_name[:-4]] = metadata
            else:
                logging.warning(f"Failed to download file for metadata: {metadata}")
                if retry > 0:
                    retries[metadata] = url
    
    if retries and retry > 0:
        logging.info(f"Retrying download for {len(retries)} files")
        await download_pdfs(retries, semaphore, visited, indexes, args, progress_report, retry)
    
    if i < args.batch: 
        reached_end_of_file = True
    
    return reached_end_of_file

async def get_base_url(url):
    """
    Extract base URL from a given full URL.
    
    Args:
        url: Full URL to extract base from
        
    Returns:
        Base URL string
    """
    if not url.startswith("http"):
        url = f"http://{url}"
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return base_url

async def setup_session(session, url, headers):
    """
    Initialize the session with base headers.
    
    Args:
        session: aiohttp ClientSession
        url: URL to initialize with
        headers: Headers to use
        
    Returns:
        Headers used for initialization
    """
    base_url = await get_base_url(url)
    initial_url = f"{base_url}"
    async with session.get(initial_url, headers=headers) as response:
        await response.text()
    return headers

async def download_pdf(index, metadata, pdf_url, semaphore, args, user_agent, referer=None):
    """
    Downloads a single PDF file asynchronously.
    
    Args:
        index: File index for naming
        metadata: Metadata for the file
        pdf_url: URL to download from
        semaphore: Semaphore for concurrency control
        args: Command-line arguments
        user_agent: User agent string
        referer: Referer URL (optional)
        
    Returns:
        Tuple of (success_status, metadata, filename)
    """
    if not referer:
        base_url = await get_base_url(pdf_url)
    else:
        base_url = referer
    
    headers = {
        'User-Agent': user_agent,
        'Referer': base_url
    }
    
    if not pdf_url.startswith("http"):
        pdf_url = f"http://{pdf_url}"
    
    sleep_time, file_type, request_type = args.sleep, args.type, args.req
    
    async with semaphore:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False), timeout=timeout) as session:
            # Randomized sleep time (better for passing bot detection)
            await asyncio.sleep(random.uniform(sleep_time, sleep_time + 2))  
           
            file_name = f'paper_{index}.{file_type}'  # Names file by order of appearance
            
            try:
                await setup_session(session, pdf_url, headers)
                requester = getattr(session, request_type)  # sets session type as either session.get or session.post
                
                async with requester(pdf_url, headers=headers, allow_redirects=False) as response:
                    if response.status in (301, 302):
                        location = response.headers.get('Location', 'unknown')
                        logging.error(f"Redirected: {pdf_url} to {location}. Status code: {response.status}")
                        return (False, metadata, file_name)
                    elif response.status == 200:
                        content = await response.read()
                        output_path = args.output
                        await write_file(file_name, content, output_path)
                        logging.info(f"Downloaded {file_name}")
                        return (True, metadata, file_name)
                    else:
                        logging.error(f"Failed to download {pdf_url}. Status code: {response.status}")
            except aiohttp.ClientError as e:
                logging.error(f"ClientError while downloading {pdf_url}: {e}")
            except aiohttp.http_exceptions.HttpProcessingError as e:
                logging.error(f"HTTP processing error while downloading {pdf_url}: {e}")
            except asyncio.TimeoutError:
                logging.error(f"Timeout error while downloading {pdf_url}")
            except Exception as e:
                logging.error(f"Unexpected error while downloading {pdf_url}: {e}")
            
            return (False, metadata, file_name)

async def write_file(filename, content, output_path="./"):
    """
    Writes downloaded content to a file asynchronously.
    
    Args:
        filename: Name of file to write
        content: Content to write
        output_path: Directory to write to
    """
    path_to_file = os.path.join(output_path, filename)
    async with aiofiles.open(path_to_file, 'wb') as file:
        await file.write(content)

def user_agent_generator():
    """
    Generator function that yields random user-agent strings to avoid bot detection.
    
    Yields:
        Random user agent string
    """
    templates = [
        "Mozilla/5.0 ({os}) AppleWebKit/537.36 (KHTML, like Gecko) {browser}/{version} Safari/537.36",
        "Mozilla/5.0 ({os}) Gecko/20100101 {browser}/{version}",
        "Mozilla/5.0 ({os}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36"
    ]
    
    operating_systems = [
        "Windows NT 10.0; Win64; x64",
        "Macintosh; Intel Mac OS X 10_15_7",
        "X11; Linux x86_64",
        "Windows NT 6.1; Win64; x64",
        "Android 9; Mobile; rv:40.0"
    ]
    
    browsers = [
        ("Chrome", random.randint(70, 90)),
        ("Firefox", random.randint(50, 80)),
        ("Edge", random.randint(80, 90))
    ]
    
    while True:
        template = random.choice(templates)
        os = random.choice(operating_systems)
        browser, version = random.choice(browsers)
        full_version = f"{version}.0.{random.randint(1000, 9999)}"
        user_agent = template.format(os=os, browser=browser, version=full_version)
        yield user_agent

async def main():
    """
    Main function to parse input arguments, load URL metadata from a JSON file or API,
    manage download progress with semaphores for concurrency, and save the download 
    progress to a JSON report file.
    """
    args = parse_input()
    semaphore = asyncio.Semaphore(3)  # Adjust the value as needed for concurrency
    metadata_dict = {}
    
    # Determine if using local JSON or API for metadata
    if args.json and os.path.exists(args.json):
        # Load metadata from JSON file
        with open(args.json, 'r') as file:
            metadata_dict = json.load(file)
        logging.info(f"Loaded metadata from JSON file: {args.json} ({len(metadata_dict)} entries)")
    elif args.api_url:
        # Fetch metadata from API with pagination
        logging.info(f"Fetching metadata from API: {args.api_url}")
        api_params = {}
        
        # Add any API-specific parameters
        if args.api_params:
            try:
                api_params = json.loads(args.api_params)
                logging.info(f"Using API parameters: {api_params}")
            except json.JSONDecodeError:
                logging.error(f"Invalid API parameters format: {args.api_params}")
        
        # Use default headers or custom headers if provided
        headers = {'User-Agent': next(user_agent_generator())}
        if args.api_headers:
            try:
                custom_headers = json.loads(args.api_headers)
                headers.update(custom_headers)
            except json.JSONDecodeError:
                logging.error(f"Invalid API headers format: {args.api_headers}")
        
        # Fetch all pages
        metadata_dict = await fetch_all_pages(args.api_url, api_params, headers, semaphore)
        
        # Save fetched metadata to a file for future use
        if metadata_dict:
            metadata_file = "fetched_metadata.json"
            with open(metadata_file, 'w') as file:
                json.dump(metadata_dict, file, ensure_ascii=False, indent=4)
            logging.info(f"Saved fetched metadata to {metadata_file}")
    else:
        logging.error("No metadata source provided. Use either --json or --api_url")
        return
    
    if not metadata_dict:
        logging.error("No metadata found. Exiting.")
        return
    
    try:
        # Read existing progress report if any
        progress_report_path = os.path.join(args.output, 'progress_report.json')
        
        try:
            with open(progress_report_path, 'r') as file:
                progress_report = json.load(file)
            logging.info(f"Existing progress report found and loaded ({len(progress_report)} entries)")
            indexes = get_indexes(list(progress_report.keys()))
        except FileNotFoundError:
            progress_report = {}
            indexes = []
            logging.info("No existing progress report found. Starting fresh.")
        
        visited = list(progress_report.values())
        
        # Start downloading PDFs
        logging.info(f"Starting download of {len(metadata_dict)} files (batch size: {args.batch})")
        finished = await download_pdfs(metadata_dict, semaphore, visited, indexes, args, progress_report)
        
        if finished:
            logging.info("All available files have been processed.")
        else:
            logging.info(f"Batch completed. Run again to continue downloading.")
    
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")
        raise
    
    finally:
        # Write progress report to a JSON file 
        with open(progress_report_path, 'w') as file:
            json.dump(progress_report, file, ensure_ascii=False, indent=4)
        logging.info(f"Progress report written to {progress_report_path}")

def parse_input():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(
        description="Advanced PDF Downloader with pagination support for academic papers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments group
    required_group = parser.add_argument_group('Required Arguments')
    source_group = required_group.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--json", help="Path to JSON file with metadata and URLs")
    source_group.add_argument("--api_url", help="API URL for fetching paginated results")
    required_group.add_argument("--type", help="File type to download (e.g., 'pdf', 'doc')", required=True)
    
    # Optional arguments
    parser.add_argument("--sleep", type=int, default=1, help="Set delay before new request is made (in seconds)")
    parser.add_argument("--req", choices=['get', 'post'], default='get', help="Set request type 'get' or 'post'")
    parser.add_argument("-o", "--output", default="./", help="Set download directory")
    parser.add_argument("--batch", type=int, default=10, help="Set number of files to download per run")
    parser.add_argument("--api_params", help="JSON string of parameters to pass to API (e.g., '{\"key\":\"value\"}')")
    parser.add_argument("--api_headers", help="JSON string of headers to pass to API (e.g., '{\"key\":\"value\"}')")
    parser.add_argument("--retry", type=int, default=3, help="Number of retry attempts for failed downloads")
    parser.add_argument("--concurrent", type=int, default=3, help="Number of concurrent downloads")
    parser.add_argument("--progress_file", default="progress_report.json", help="Name of the progress tracking file")
    parser.add_argument("--log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        default='INFO', help="Set logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        logging.info(f"Created output directory: {args.output}")
    
    logging.info(f"Arguments received: JSON file: {args.json}, API URL: {args.api_url}, "
                 f"Sleep time: {args.sleep}, File type: {args.type}, Request type: {args.req}, "
                 f"Output path: {args.output}, Batch size: {args.batch}")
    
    return args

#Entry point of Downloader 
if __name__ == "__main__":
    asyncio.run(main())