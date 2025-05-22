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


# Configure logging for behavior tracking and errors 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function for the highest index of papers downloaded for continuation
def get_indexes(papers):
    if papers:
        nums = []
        for p in papers:
            # Extract just the index number from paper_X_... format
            parts = p.split("_")
            if len(parts) > 1 and parts[0] == "paper":
                try:
                    nums.append(int(parts[1]))
                except ValueError:
                    logging.warning(f"Could not parse index from {p}, skipping...")
        return sorted(nums)[-1:] if nums else [0]
    return [0]  # Start from index 1 if no papers

# Function that is capable of downloading PDFs allowing retrial and concurrent downloads 
async def download_pdfs(metadata_dict, semaphore, visited, indexes, args, progress_report, retry=3):
    """Prepares tasks for download_pdf function and manages retries for failed downloads."""
    
    retry -= 1
    retries = {}  # Dictionary holding files for download retrial
    tasks = []  # List to hold the tasks to be executed
    ordered_metadata = list(metadata_dict.items())
    user_agent_gen = user_agent_generator()
    i = 0
    reached_end_of_file = True  # flag: if all metadata are in "visited"
    
    # Process metadata urls and schedule downloads
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
        await download_pdfs(retries, semaphore, visited, indexes, args, progress_report, retry-1)
    if i < args.batch: reached_end_of_file = True
    return reached_end_of_file

# Function to extract base URL from a given full URL
async def get_base_url(url):
    if not url.startswith("http"):
        url = f"https://{url}"
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return base_url

# Function for the initialization of session headers
async def setup_session(session, url, headers):
    """ Initialize the session with base headers. """
    base_url = await get_base_url(url)
    initial_url = f"{base_url}"
    try:
        async with session.get(initial_url, headers=headers, ssl=False, timeout=30) as response:
            await response.text()
    except Exception as e:
        logging.warning(f"Session setup failed, but continuing: {e}")
    return headers

# Function that arranges concurrent download of PDFs given pdf_url
async def download_pdf(index, metadata, pdf_url, semaphore, args, user_agent, referer=None):
    """Downloads a PDF file and returns a tuple of (success, metadata, filename)."""

    if not referer:
        base_url = await get_base_url(pdf_url)
    else:
        base_url = referer
        
    # Enhanced headers to mimic a real browser - specific for greek-language.gr
    headers = {
        'User-Agent': user_agent,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'el-GR,el;q=0.9,en-US;q=0.8,en;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.greek-language.gr/',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'same-origin',
        'Pragma': 'no-cache',
        'Cache-Control': 'no-cache',
        'TE': 'trailers'
    }
    
    # Handle HTTPS URLs properly
    if not pdf_url.startswith("http"):
        pdf_url = f"https://{pdf_url}"
        
    sleep_time, file_type, request_type = args.sleep, args.type, args.req
    
    async with semaphore:
        # Use a reasonable timeout
        timeout = aiohttp.ClientTimeout(total=60)
        
        # Sleep between requests - use moderate sleep for greek-language.gr to avoid overloading
        await asyncio.sleep(random.uniform(sleep_time, sleep_time + 2))
           
        # Create a unique filename based on the metadata for better organization
        # Replace problematic characters in filename
        safe_metadata = metadata.replace("/", "_").replace("\\", "_").replace(":", "_").strip()
        file_name = f'paper_{index}_{safe_metadata}.{file_type}'
        
        try:
            # Setup a session with SSL verification disabled
            connector = aiohttp.TCPConnector(ssl=False, force_close=False)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                # Try to visit the main site first to set cookies
                try:
                    await setup_session(session, base_url, headers)
                except Exception as e:
                    logging.warning(f"Base site visit failed, continuing: {e}")
                
                # For greek-language.gr, GET should work 
                requester = getattr(session, request_type)
                try:
                    # Allow redirects
                    async with requester(pdf_url, headers=headers, allow_redirects=True, max_redirects=5) as response:
                        if response.status == 200:
                            content = await response.read()
                            # Check if content is valid PDF (starts with %PDF)
                            if content[:4].startswith(b'%PDF'):
                                if args.output: 
                                    output_path = args.output
                                    # Ensure the output directory exists
                                    os.makedirs(output_path, exist_ok=True)
                                await write_file(file_name, content, output_path)
                                logging.info(f"Downloaded {file_name}")
                                return (True, metadata, file_name)
                            else:
                                logging.error(f"Downloaded content is not a valid PDF: {pdf_url}")
                                return (False, metadata, file_name)
                        elif response.status in (301, 302, 303, 307, 308):
                            # Log redirection but don't treat as failure since we're following redirects
                            location = response.headers.get('Location', 'Unknown')
                            logging.info(f"Following redirect: {pdf_url} to {location}")
                            return (False, metadata, file_name)
                        else:
                            logging.error(f"Failed to download {pdf_url}. Status code: {response.status}")
                            return (False, metadata, file_name)
                except aiohttp.ClientError as e:
                    logging.error(f"ClientError while downloading {pdf_url}: {e}")
                except Exception as e:
                    logging.error(f"Error downloading {pdf_url}: {e}")
                        
            return (False, metadata, file_name)
                                
        except aiohttp.ClientError as e:
            logging.error(f"ClientError while downloading {pdf_url}: {e}")
        except aiohttp.http_exceptions.HttpProcessingError as e:
            logging.error(f"HTTP processing error while downloading {pdf_url}: {e}")
        except asyncio.TimeoutError:
            logging.error(f"Timeout error while downloading {pdf_url}")
        except Exception as e:
            logging.error(f"Unexpected error while downloading {pdf_url}: {e}")
        return (False, metadata, file_name)

# Function that writes downloaded content to a file 
async def write_file(filename, content, output_path = "./"):
    path_to_file = os.path.join(output_path, filename)
    async with aiofiles.open(path_to_file, 'wb') as file:
        await file.write(content)

# Function to generate random user-agents for avoiding bot detection 
def user_agent_generator():
    """Generates realistic user agent strings for requests."""
    
    templates = [
        "Mozilla/5.0 ({os}) AppleWebKit/537.36 (KHTML, like Gecko) {browser}/{version} Safari/537.36",
        "Mozilla/5.0 ({os}) Gecko/20100101 {browser}/{version}",
        "Mozilla/5.0 ({os}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36 {browser}/{version}"
    ]
    operating_systems = [
        "Windows NT 10.0; Win64; x64",
        "Macintosh; Intel Mac OS X 10_15_7",
        "X11; Linux x86_64",
        "Windows NT 6.1; Win64; x64",
        "Android 9; Mobile; rv:40.0"
    ]
    browsers = [
        ("Chrome", random.randint(90, 120)),  # Updated versions
        ("Firefox", random.randint(90, 120)),
        ("Edge", random.randint(90, 120))
    ]
    while True:
        template = random.choice(templates)
        os = random.choice(operating_systems)
        browser, version = random.choice(browsers)
        full_version = f"{version}.0.{random.randint(1000, 9999)}"
        user_agent = template.format(os=os, browser=browser, version=full_version)
        yield user_agent

# Function for overall program execution 
async def run(args):
    """Main execution function that loads metadata and manages the download process."""
    
    current_working_directory = os.getcwd()
    path_to_url_siteguide = os.path.join(current_working_directory, args.json)
    with open(path_to_url_siteguide, 'r') as file:
        metadata_dict = json.load(file)

    # For greek-language.gr, use lower concurrency to avoid being blocked
    semaphore = asyncio.Semaphore(1)  # Only 1 concurrent request
    try:
        # Load existing progress or create new
        try:
            progress_report_path = os.path.join(args.little_potato, 'progress_report.json')
            with open(progress_report_path, 'r') as file:
                progress_report = json.load(file)
            logging.info("Existing progress report found and loaded")
            indexes = get_indexes(list(progress_report.keys()))
        except FileNotFoundError:
            progress_report = {}
            indexes = [0]  # Start with index 1
            logging.info("No existing progress report found")
        except Exception as e:
            # If there's an error reading the progress report, start fresh
            logging.error(f"Error reading progress report: {e}")
            progress_report = {}
            indexes = [0]  # Start with index 1
        
        visited = list(progress_report.values())
        logging.info(f"Starting download from {args.json}")
        finished = await download_pdfs(metadata_dict, semaphore, visited, indexes, args, progress_report)
        logging.info(f"Finished download from {args.json}")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        finished = False
    finally:
        # Always save progress
        if 'finished' in locals() and finished:
            logging.info("All available files have been downloaded - Finished!")
        else:
            logging.info("PDF downloads completed for this batch")
        
        progress_report_path = os.path.join(args.little_potato, 'progress_report.json')
        with open(progress_report_path, 'w') as file:
            json.dump(progress_report, file, ensure_ascii=False, indent=4)
        logging.info("Progress report written to progress_report.json")
        return 'finished' in locals() and finished

# Function for handling command-line arguments 
def parse_input():
    """Parses and validates command line arguments."""
    
    parser = argparse.ArgumentParser(description="Downloads PDFs from greek-language.gr", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--json", help="Add path to JSON file with URLs siteguide", required=True)
    parser.add_argument("--sleep", type=int, default=3, help="Set delay before new request is made (in seconds)")
    parser.add_argument("--type", help="Select file type to be downloaded e.g., 'pdf', 'doc'", required=True)
    parser.add_argument("--req", choices=['get', 'post'], default='get', help="Set request type 'get' or 'post'")
    parser.add_argument("-o", "--output", default="./", help="Set download directory")
    parser.add_argument("--little_potato", help="Set directory for progress_report.json (previously little_potato), default value is set to --output")
    parser.add_argument("--batch", type=int, default=5, help="Set number of files to download per run")
    args = parser.parse_args()

    if not args.little_potato:
        args.little_potato = args.output
    logging.info(f"Arguments received: JSON file: {args.json}, Sleep time: {args.sleep}, File type: {args.type}, Request type: {args.req}, Output path: {args.output}, 'progress_report.json' path: {args.little_potato}")
    return args

# Entry point of Downloader 
if __name__ == "__main__":
    args = parse_input()
    asyncio.run(run(args))
