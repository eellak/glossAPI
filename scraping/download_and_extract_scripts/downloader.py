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
            num = p.split("_")[-1]
            nums.append(int(num))
        return sorted(nums)[-1:]
    return []

# Function that is capable of downloading PDFs allowing retrial and concurrent downloads 
async def download_pdfs(metadata_dict, semaphore, visited, indexes, args, progress_report, retry=1):
    retry -= 1
    retries = {}  # Dictionary holding files for download retrial
    tasks = []  # List to hold the tasks to be executed
    ordered_metadata = list(metadata_dict.items())
    user_agent_gen = user_agent_generator()
    i = 0
    reached_end_of_file = True  # flag: if all metadata are in "visited"
    
    for metadata, url in ordered_metadata:
        if i < args.batch and metadata not in visited:
            reached_end_of_file = False
            index = indexes[-1] + 1 if indexes else 1
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
                    retries[url] = metadata
    if retries and retry > 0:
        logging.info(f"Retrying download for {len(retries)} files")
        await download_pdfs(retries, semaphore, visited, indexes, args, progress_report, retry-1)
    if i < args.batch: 
        reached_end_of_file = True
    return reached_end_of_file

# Function to extract base URL from a given full URL
async def get_base_url(url):
    if not url.startswith("http"):
        url = f"http://{url}"
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return base_url

# Function for the initialization of session headers
async def setup_session(session, url, headers):
    base_url = await get_base_url(url)
    initial_url = f"{base_url}"
    async with session.get(initial_url, headers=headers) as response:
        await response.text()
    return headers

# Function that arranges concurrent download of PDFs
async def download_pdf(index, metadata, pdf_url, semaphore, args, user_agent, referer=None):
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
            await asyncio.sleep(random.uniform(sleep_time, sleep_time + 2))  
           
            file_name = f'paper_{index}.{file_type}'
            try:
                await setup_session(session, pdf_url, headers)
                requester = getattr(session, request_type)
                async with requester(pdf_url, headers=headers, allow_redirects=False) as response:
                    if response.status in (301, 302):
                        logging.error(f"Redirected: {pdf_url} to {response.headers['Location']}. Status code: {response.status}")
                        return (False, metadata, file_name)
                    elif response.status == 200:
                        content = await response.read()
                        if args.output: output_path = args.output
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

# Function that writes downloaded content to a file 
async def write_file(filename, content, output_path = "./"):
    path_to_file = os.path.join(output_path, filename)
    async with aiofiles.open(path_to_file, 'wb') as file:
        await file.write(content)

# Function to generate random user-agents
def user_agent_generator():
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

# Function for overall program execution 
async def run(args):
    current_working_directory = os.getcwd()
    path_to_url_siteguide = os.path.join(current_working_directory, args.filename)
    with open(path_to_url_siteguide, 'r') as file:
        metadata_dict = json.load(file)

    semaphore = asyncio.Semaphore(args.concurrency)  # Use concurrency argument for semaphore limit
    try:
        try:
            with open('progress_report.json', 'r') as file:
                progress_report = json.load(file)
            logging.info("Existing progress report found and loaded")
            indexes = get_indexes(list(progress_report.keys()))
        except FileNotFoundError:
            progress_report = {}
            indexes = []
            logging.info("No existing progress report found")
        visited = list(progress_report.values())
        logging.info(f"Starting download from {args.filename}")
        finished = await download_pdfs(metadata_dict, semaphore, visited, indexes, args, progress_report)
        logging.info(f"Finished download from {args.filename}")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
    finally:
        if finished:
            logging.info("All available have been downloaded - Finished!")
            with open('progress_report.json', 'w') as file:
                json.dump(progress_report, file, ensure_ascii=False, indent=4)
            return True
        else:
            logging.info("PDF downloads completed")
            with open('progress_report.json', 'w') as file:
                json.dump(progress_report, file, ensure_ascii=False, indent=4)
            logging.info("Progress report written to progress_report.json")
            return False

# Function for handling command-line arguments 
def parse_input():
    parser = argparse.ArgumentParser(description="Gets PDFs through URLs given as value entries in a JSON.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--json", help="Add path to JSON file with URLs siteguide", required=True)
    parser.add_argument("--sleep", type=int, default=1, help="Set delay before new request is made (in seconds)")
    parser.add_argument("--type", help="Select file type to be downloaded e.g., 'pdf', 'doc'", required=True)
    parser.add_argument("--req", choices=['get', 'post'], default='get', help="Set request type 'get' or 'post'")
    parser.add_argument("-o", "--output", default="./", help="Set download directory")
    parser.add_argument("--little_potato", help="Set directory for progress_report.json (previously little_potato_downloaded)")
    parser.add_argument("--concurrency", type=int, default=3, help="Set the concurrency limit (number of concurrent downloads)")
    parser.add_argument("--batch", type=int, default=5, help="Set number of files to download per run")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_input()
    asyncio.run(run(args))
