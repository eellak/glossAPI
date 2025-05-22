# Cyprus Exams Downloader

This module is responsible for downloading examination papers and educational materials from the Cyprus Ministry of Education website.

## Data Description

The data we download includes:
- Pan-Cyprian Examination Papers
- Sample examination papers
- Solutions and answers to examination papers
- Educational materials and guidelines

These files provide important educational material for students, pupils, and educators in Cyprus, and form part of the country's official education system.

## Scripts

Data downloading is performed using the following scripts:

1. **downloader_cyprus_exams.py**: The main script that downloads PDFs from Cyprus examinations.
2. **download_all_cyprus_exams.sh**: A shell script that automates the downloading process for all files.
3. **monitor_cyprus_exams.py**: A script that monitors the download progress.

## Technical Implementation

The downloader_cyprus_exams.py script has the following characteristics:

- **Flexible JSON processing**: Can handle both URL lists and complex JSON objects with keys and values.
- **Asynchronous operation**: Uses `asyncio` and `aiohttp` libraries for efficient downloading.
- **Rate limiting**: Uses semaphore to limit the number of concurrent connections to 1, thus avoiding server overload.
- **Polite behavior**: Adds delays between requests to avoid server overload.
- **Robustness**: Handles errors and automatically retries on failure.
- **Progress tracking**: Saves progress to a JSON file for continuation from breakpoint.
- **Data validation**: Checks if downloaded files are valid PDFs.

### Key Features

1. **Special character handling**: Properly handles URLs containing special characters.
2. **Custom HTTP headers**: Uses headers that mimic a real browser for better compatibility.
3. **Error retries**: Attempts up to 3 times with increasing wait time on failure.
4. **PDF validity check**: Checks the first bytes of downloaded file to confirm it's a valid PDF.

## Usage Instructions

### Prerequisites

```bash
pip install aiohttp aiofiles
```

### Download all files

```bash
# Create directory for files
mkdir -p ../../downloads/cyprus-exams

# Execute automation script
bash download_all_cyprus_exams.sh
```

### Monitor progress

```bash
python3 monitor_cyprus_exams.py
```

### Manual execution

```bash
python3 downloader_cyprus_exams.py --json ../../scraping/json_sitemaps/cyprus-exams_pdf.json --type pdf --req get --output ../../downloads/cyprus-exams --batch 5 --sleep 3
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

### 1. Handling different URL formats

The script handles different URL formats, including those containing special characters or encoded parameters.

### 2. Avoiding server blocking

The script uses only 1 concurrent connection and waits approximately 3 seconds between requests to avoid server blocking.

### 3. Handling large files

The script uses streaming to download files in chunks, thus avoiding loading the entire file into memory.

## Results

The script downloads files from the Cyprus Ministry of Education website and stores them in the `downloads/cyprus-exams/` directory. These files are important for educational research and provide valuable material for analyzing the educational process in Cyprus.

## Technical Specifications

- **Concurrent downloads**: 1 simultaneous connection  
- **Request delay**: 3 seconds between requests
- **Timeout**: 60 seconds per request
- **Retry strategy**: 3 retries with 5-second delays
- **SSL verification**: Enabled for security
- **Greek language support**: Specialized headers for Cyprus sites
- **PDF validation**: Content signature verification
- **Filename sanitization**: Filesystem-safe filename generation