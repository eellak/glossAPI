# Panhellenic Examinations Downloader

This module is responsible for downloading examination papers and solutions from the Panhellenic Examinations in Greece.

## Data Description

The data we download includes:
- Panhellenic Examination papers from 2001 to present
- Solutions and answers to the papers
- Files for various subjects (Modern Greek Language, Mathematics, Physics, History, etc.)

These files provide important educational material for students, pupils, and educators in Greece, and form part of the country's official education system.

## Scripts

Data downloading is performed using the following scripts:

1. **downloader_panelladikes.py**: The main script that downloads PDFs with Panhellenic Examination papers and solutions.
2. **download_all_panelladikes.sh**: A shell script that automates the downloading process for all files.
3. **monitor_panelladikes.py**: A script that monitors the download progress.

## Technical Implementation

The downloader_panelladikes.py script has the following characteristics:

- **Flexible JSON processing**: Can handle both URL lists and complex JSON objects with keys and values.
- **Asynchronous operation**: Uses `asyncio` and `aiohttp` libraries for efficient downloading.
- **Rate limiting**: Uses semaphore to limit the number of concurrent connections to 1, thus avoiding server overload.
- **Polite behavior**: Adds delays between requests to avoid server overload.
- **Robustness**: Handles errors and automatically retries on failure.
- **Progress tracking**: Saves progress to a JSON file for continuation from breakpoint.
- **Data validation**: Checks if downloaded files are valid PDFs.

### Key Features

1. **Special character handling**: Properly handles file titles containing special characters.
2. **Custom HTTP headers**: Uses headers that mimic a real browser for better compatibility.
3. **Error retries**: Attempts up to 3 times with exponential backoff on failure.
4. **PDF validity check**: Checks the first bytes of downloaded file to confirm it's a valid PDF.
5. **Professional logging**: Dual logging to console and file with detailed error tracking.

## Usage Instructions

### Prerequisites

```bash
pip install aiohttp aiofiles
```

### Download all files

```bash
# Create directory for files
mkdir -p ../../downloads/panelladikes-exams

# Execute automation script
bash download_all_panelladikes.sh
```

### Monitor progress

```bash
python3 monitor_panelladikes.py
```

### Manual execution

```bash
python3 downloader_panelladikes.py --json ../../scraping/json_sitemaps/themata-lyseis-panelladikwn_pdf.json --type pdf --req get --output ../../downloads/panelladikes-exams --batch 5 --sleep 3
```

## Parameters

- `--json`: Path to JSON file with URLs
- `--type`: Type of files to download (e.g., 'pdf')
- `--req`: HTTP request type ('get' or 'post')
- `--output`: Download directory for downloaded files
- `--batch`: Number of files per execution (default: 5)
- `--sleep`: Delay between requests in seconds (default: 3)
- `--little_potato`: Directory for progress files

## Challenges and Solutions

### 1. Handling different title formats

The script handles complex file titles (e.g., "2023 > History > Papers") by converting them to safe filenames by removing special characters.

### 2. Avoiding server blocking

The script uses only 1 concurrent connection and waits approximately 3 seconds between requests to avoid server blocking.

### 3. Handling malformed URLs

Some URLs in the JSON file contain errors, such as double domains (e.g., "https://eduadvisor.grhttp://www.minedu.gov.gr/..."). The script detects these cases and:

1. Fixes URLs by replacing the erroneous prefix "eduadvisor.grhttp" with "http".
2. Completely skips known problematic URLs, such as certain 2013 and 2011 files.
3. Logs all actions in detail for easy monitoring.

These fixes significantly increase the download success rate.

## Results

The script downloads Panhellenic Examination files from the Ministry of Education website and other reliable sources and stores them in the `downloads/panelladikes-exams/` directory. These files are important for educational research and provide valuable material for analyzing the educational process in Greece.

## Recent Execution Results

During the most recent script execution (21/05/2025), the following results were observed:

```
2025-05-21 23:03:21,936 - ERROR - Failed to download https://eduadvisor.gr/images/stories/pdf/ΠΑΝΕΛΛΗΝΙΕΣ%202013/ΘΕΜΑΤΑ%20ΚΑΙ%20ΑΠΑΝΤΗΣΕΙΣ/2001-2011/ΤΕΧΝΟΛΟΓΙΚΗ%20ΚΑΤΕΥΘΥΝΣΗ/Φυσική%20Κατεύθυνσης.pdf after 3 attempts
2025-05-21 23:03:21,938 - INFO - Download complete. Summary: 2 successful, 99 already existed, 26 failed
```

Out of 127 total files:
- 2 downloaded successfully
- 99 already existed locally
- 26 failed to download

We observe that despite improvements made to the script, some URLs still cannot be downloaded, mainly because the files no longer exist on the respective servers (HTTP 404). The majority of files (>78%) were either successfully downloaded or already existed locally, showing that the script works effectively for most files.

## Technical Specifications

- **Concurrent downloads**: 1 simultaneous connection
- **Request delay**: 3 seconds between requests
- **Timeout**: 60 seconds per request
- **Retry strategy**: 3 retries with exponential backoff
- **SSL verification**: Enabled for security
- **URL validation**: Automatic fixing of malformed URLs
- **Dual logging**: Console and file logging
- **PDF validation**: Content signature verification