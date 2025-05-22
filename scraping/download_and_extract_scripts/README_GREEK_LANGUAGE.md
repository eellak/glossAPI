# Greek Language Downloader

This module is responsible for downloading educational materials from [greek-language.gr](https://www.greek-language.gr/), an important resource for the Greek language and literature.

## Data Description

The data we download from greek-language.gr includes:
- KGL (Text-Language Function) exercises
- Practice materials for various levels (A1, A2)
- KPL (Oral Speech Comprehension) texts
- Exercise solutions

The collection consists of approximately 50 PDFs containing educational materials for teaching and learning the Greek language.

## Scripts

Data downloading is performed using the following scripts:

1. **downloader_greek_language.py**: The main script that downloads PDFs from greek-language.gr.
2. **download_all_greek_language.sh**: A shell script that automates the downloading process for all files.
3. **monitor_greek_language.py**: A script that monitors the download progress.

## Technical Implementation

The downloader_greek_language.py script has the following characteristics:

- **Asynchronous operation**: Uses `asyncio` and `aiohttp` libraries for efficient downloading.
- **Polite behavior**: Limits concurrent requests to 1 and adds delays between requests.
- **Robustness**: Handles errors and automatically retries on failure.
- **Progress tracking**: Saves progress to a JSON file for continuation from breakpoint.
- **Data validation**: Checks if downloaded files are valid PDFs.

### Key Features

1. **Greek character handling**: Properly handles URLs and filenames with Greek characters.
2. **Custom HTTP headers**: Mimics a browser from Greece for better compatibility.
3. **Overload avoidance**: Uses low parallel request limit (1) to avoid blocking.
4. **Progress monitoring**: Maintains a record of downloaded files for restart capability.

## Usage Instructions

### Prerequisites

```bash
pip install aiohttp aiofiles
```

### Download all files

```bash
# Create directory for files
mkdir -p ../../downloads/greek-language

# Execute automation script
bash download_all_greek_language.sh
```

### Monitor progress

```bash
python3 monitor_greek_language.py
```

### Manual execution

```bash
python3 downloader_greek_language.py --json ../../scraping/json_sitemaps/greek-language_pdf.json --type pdf --req get --output ../../downloads/greek-language --batch 5 --sleep 3
```

## Parameters

- `--json`: Path to JSON file with URLs
- `--type`: Type of files to download (e.g., 'pdf')
- `--req`: HTTP request type ('get' or 'post')
- `--output`: Download directory for downloaded files
- `--batch`: Number of files per execution (default: 5)
- `--sleep`: Delay between requests in seconds (default: 3)
- `--little_potato`: Directory for progress_report.json file

## Challenges and Solutions

### 1. Handling Greek characters in URLs

URLs contain Greek characters that can cause encoding problems. The solution was using the `aiohttp` library which properly handles URL encoding.

### 2. Avoiding server blocking

The script uses only 1 concurrent connection and waits approximately 3 seconds between requests to avoid server blocking.

### 3. Downloaded file validation

The script checks if downloaded files are actually PDFs by examining the first bytes for the "%PDF" signature.

### 4. Progress monitoring

A separate script was implemented for real-time progress monitoring.

## Results

The script successfully downloaded all files from greek-language.gr. The files are available in the `downloads/greek-language/` directory.

## Technical Specifications

- **Concurrent downloads**: 1 simultaneous connection
- **Request delay**: 3-5 seconds between requests
- **Timeout**: 60 seconds per request
- **Retry strategy**: 3 retries with progressive delays
- **SSL verification**: Disabled for compatibility
- **Default protocol**: HTTPS
- **Greek language support**: Specialized headers for Greek sites
- **PDF validation**: Content signature verification