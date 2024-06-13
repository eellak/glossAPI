import json
import asyncio
import argparse
import aiohttp
import aiofiles
import random
import logging
from urllib.parse import urlparse

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main():
    setup_logging()
    args = parse_input()
    metadata_dict = await read_json_to_dict(args.json)
    logging.info(f"Length of files is {len(metadata_dict)}")
    semaphore = asyncio.Semaphore(3)
    progress = Progress()
    await progress.get_progress
    async with progress:
        try:
            await download_pdfs(metadata_dict, semaphore, progress.file, progress.visited, progress.indexes, args)
        except Exception as e:
            logging.error(f"Didn't complete list due to an error: {str(e)}")
            raise RuntimeError(f"Didn't complete list due to an error: {str(e)}")

class Progress():
    """
    Sets up context manager for writing to JSON;
    also finds and continues from interrupted operation
    if it exists.
    """
    def __init__(self):
        self.indexes = []
        self.visited = []
        self.filename = "little_potato.json"
        self.file = None
        self.progress = {}

    @property
    async def get_progress(self):
        try:
            self.progress = await read_json_to_dict(f"./{self.filename}")
            logging.info("Progress file detected and read")
            if len(self.progress) > 1:
                papers = list(self.progress.keys())
                self.indexes = self.get_indexes(papers)
                self.visited = list(self.progress.values())
        except FileNotFoundError:
            logging.info("No progress file found")
        except Exception as e:
            logging.error(f"Error reading progress file: {str(e)}")

    async def __aenter__(self):
        self.file = await aiofiles.open(self.filename, 'w', encoding='utf-8')
        await self.file.write('{\n')
        if self.progress:
            for paper_num, metadata in self.progress.items():
                await self.file.write(json.dumps(paper_num, ensure_ascii=False, indent=4) + ': ' + json.dumps(metadata, ensure_ascii=False, indent=4) + ',\n')

    async def __aexit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            if exc_type is RuntimeError:
                logging.error(f"RuntimeError occurred: {exc_value}")
            else:
                logging.error(f"An error occurred: {exc_value}")
        await self.file.write('\n}')
        await self.file.close()

    def get_indexes(self, papers):
        if papers:
            nums = []
            for p in papers:
                num = p.replace("paper_", "")
                nums.append(int(num))
            return sorted(nums)[-1:]
        return []

async def read_json_to_dict(filename):
    async with aiofiles.open(filename, 'r', encoding='utf-8') as file:
        data = await file.read()
    return json.loads(data)

async def download_pdfs(metadata_dict, semaphore, file, visited, indexes, args, retry=1):
    """
    Prepares tasks for download_pdf function; and stores association of 
    "paper_n.pdf" name with original metadata.
    """
    retry -= 1
    retries = {}
    tasks = []
    ordered_metadata = list(metadata_dict.items())
    user_agent_gen = user_agent_generator()

    i = 0
    for metadata, pdf_url in ordered_metadata:
        if i < 10 and metadata not in visited:
            if indexes:
                index = indexes[-1] + 1
            else:
                index = 1
            indexes.append(index)
            task = asyncio.create_task(
                download_pdf(index, metadata, pdf_url, semaphore, args, next(user_agent_gen))
            )
            tasks.append(task)
            i += 1

    first = True

    results = await asyncio.gather(*tasks)
    for result in results:
        if result:
            hasFile, metadata, pdf_file_name = result
            if hasFile:
                logging.info(f"Downloaded {pdf_file_name}")
                if not first:
                    await file.write(',\n')
                first = False
                await file.write(json.dumps(pdf_file_name[:-4], ensure_ascii=False, indent=4) + ': ' + json.dumps(metadata, ensure_ascii=False, indent=4))
            else:
                if retry > 0:
                    retries[metadata] = pdf_url

    if retries and retry > 0:
        await download_pdfs(metadata_dict, semaphore, file, visited, indexes, args, retry-1)

async def get_base_url(url):
    if not url.startswith("http"):
        url = f"http://{url}"
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return base_url

async def setup_session(session, url, headers):
    """ Initialize the session with base headers. """
    base_url = await get_base_url(url)
    initial_url = f"{base_url}"
    async with session.get(initial_url, headers=headers) as response:
        await response.text()
    return headers

async def download_pdf(index, metadata, pdf_url, semaphore, args, user_agent, referer=None):
    """
    Arranges download of a PDF given pdf_url concurrently, then returns
    metadata and name given to downloaded PDF as a tuple.
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
        async with aiohttp.ClientSession() as session:
            await asyncio.sleep(int(sleep_time))  # Delay before each request
            pdf_file_name = f'paper_{index}.{file_type}'  # Name PDFs by order of appearance
            await setup_session(session, pdf_url, headers)
            requester = getattr(session, request_type)  # sets session type as either session.get or session.post
            try:
                async with requester(pdf_url, headers=headers) as response:
                    if response.status == 200:
                        content = await response.read()
                        await write_file(pdf_file_name, content)
                        return (True, metadata, pdf_file_name)
                    else:
                        logging.error(f"Failed to download {pdf_url}")
            except aiohttp.ClientError as e:
                logging.error(f"ClientError while downloading {pdf_url}: {e}")
            except aiohttp.http_exceptions.HttpProcessingError as e:
                logging.error(f"HTTP processing error while downloading {pdf_url}: {e}")
            except asyncio.TimeoutError:
                logging.error(f"Timeout error while downloading {pdf_url}")
            except Exception as e:
                logging.error(f"Unexpected error while downloading {pdf_url}: {e}")
            return (False, metadata, pdf_file_name)

async def write_file(filename, content):
    async with aiofiles.open(filename, 'wb') as file:
        await file.write(content)

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

def parse_input():
    parser = argparse.ArgumentParser(description="Gets PDFs through URLs given as value entries in a JSON.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--json", help="Add path to .json file", required=True)
    parser.add_argument("--sleep", type=int, default=1, help="Set delay before new request is made in seconds")
    parser.add_argument("--type", help="Select file type to be downloaded e.g., 'pdf', 'doc'", required=True)
    parser.add_argument("--req", choices=['get', 'post'], default='get', help="Set request type 'get' or 'post'")
    args = parser.parse_args()
    logging.info(f"Arguments received: JSON file: {args.json}, Sleep time: {args.sleep}, File type: {args.type}, Request type: {args.req}")
    return args

if __name__ == "__main__":
    asyncio.run(main())
