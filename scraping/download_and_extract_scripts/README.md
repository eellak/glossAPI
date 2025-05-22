# GlossAPI Download Scripts Collection

## Overview

This directory contains specialized downloader scripts for various Greek educational and legal document repositories. Each script is tailored to the specific requirements and characteristics of different websites, based on the original `downloader.py` by Nikos Tsekos.

## Repository Status Summary

| Repository | Script | Status | Success Rate | Files Downloaded | Notes |
|------------|--------|---------|--------------|------------------|--------|
| **Kodiko** | `downloader_kodiko.py` | ✅ **Completed** | **86.95%** | **23,086/26,552** | Excellent performance |
| **Greek Language** | `downloader_greek_language.py` | ✅ **Completed** | **~100%** | **~50 files** | All files successfully downloaded |
| **Cyprus Exams** | `downloader_cyprus_exams.py` | ✅ **Active** | **Good** | **Variable** | Ongoing downloads |
| **Panelladikes** | `downloader_panelladikes.py` | ✅ **Active** | **~78%** | **101/127** | URL issues resolved |
| **Kallipos** | `downloader_kallipos.py` | ❌ **Blocked** | **0%** | **0** | HTTP 500 errors persist |

## Script Comparison Table

| Feature | Original | Cyprus | Greek Lang | Panelladikes | Kodiko | Kallipos |
|---------|----------|---------|------------|--------------|---------|----------|
| **Concurrent Downloads** | 3 | 1 | 1 | 1 | 2 | 1 |
| **Default Sleep (sec)** | 1 | 3 | 3 | 3 | 1 | 1 |
| **Max Sleep Range** | +2 | - | +2 | - | +3 | +5 |
| **Timeout (sec)** | 60 | 60 | 60 | 60 | 60 | 120 |
| **SSL Verification** | Disabled | Enabled | Disabled | Enabled | Disabled | Disabled |
| **HTTPS Default** | No | No | Yes | Yes | Yes | Yes |
| **PDF Validation** | No | Yes | Yes | Yes | No | No |
| **Metadata in Filename** | No | Yes | Yes | No | No | No |
| **Multiple HTTP Methods** | No | No | No | No | No | Yes |
| **Exponential Backoff** | No | No | No | Yes | No | No |
| **URL Fixing** | No | No | No | Yes | No | No |
| **Greek Headers** | No | Yes | Yes | No | No | No |
| **File Logging** | No | No | No | Yes | No | No |

## Individual Script Documentation

### 1. Kodiko Downloader (`downloader_kodiko.py`)
- **Purpose**: Legal documents from kodiko.gr
- **Status**: ✅ **Excellent Success (86.95%)**
- **Files**: 23,086 out of 26,552 downloaded
- **Approach**: Moderate concurrency (2), balanced sleep times
- **Special Features**: Google referer, comprehensive headers

### 2. Greek Language Downloader (`downloader_greek_language.py`)
- **Purpose**: Educational materials from greek-language.gr
- **Status**: ✅ **Complete Success**
- **Files**: ~50 PDF files (all downloaded)
- **Approach**: Ultra-conservative (1 concurrent), Greek-specific headers
- **Special Features**: Greek character handling, PDF validation

### 3. Cyprus Exams Downloader (`downloader_cyprus_exams.py`)
- **Purpose**: Examination papers from Cyprus Ministry of Education
- **Status**: ✅ **Active Downloads**
- **Approach**: Conservative (1 concurrent), robust error handling
- **Special Features**: Greek language support, filename sanitization

### 4. Panelladikes Downloader (`downloader_panelladikes.py`)
- **Purpose**: Panhellenic Examination papers
- **Status**: ✅ **Good Success (~78%)**
- **Files**: 101 out of 127 successfully processed
- **Approach**: Professional structure, exponential backoff
- **Special Features**: URL fixing, dual logging, malformed URL detection

### 5. Kallipos Downloader (`downloader_kallipos.py`)
- **Purpose**: Academic textbooks from Kallipos repository
- **Status**: ❌ **Blocked (HTTP 500 errors)**
- **Files**: 0 (server-side blocking)
- **Approach**: Ultra-conservative, multiple methods, extended timeouts
- **Special Features**: GET/POST fallback, longest timeouts, max redirects

## Automation Scripts

Each downloader comes with supporting shell scripts for automation:

### Shell Scripts (`.sh` files)
- `download_all_kodiko.sh`
- `download_all_greek_language.sh`
- `download_all_cyprus_exams.sh`
- `download_all_panelladikes.sh`
- `download_all_kallipos.sh` (currently non-functional due to server blocking)

### Monitor Scripts (`monitor_*.py`)
Real-time progress monitoring for each repository:
- `monitor_kodiko.py`
- `monitor_greek_language.py`
- `monitor_cyprus_exams.py`
- `monitor_panelladikes.py`
- `monitor_kallipos.py`

## Common Features

All specialized downloaders maintain the core functionality of the original:

1. **Asynchronous downloads** using `asyncio` and `aiohttp`
2. **Progress tracking** with JSON progress reports  
3. **Command-line interface** with argparse
4. **Error handling** and retry mechanisms
5. **User agent rotation** for bot detection avoidance
6. **Semaphore-based concurrency control**
7. **Configurable batch sizes and sleep times**

## Site-Specific Optimizations

### Browser Simulation
- **Headers**: Modern browser headers with Greek language support
- **User Agents**: Updated browser versions (90-120 vs original 70-90)
- **Referers**: Site-appropriate referer headers
- **Security**: Sec-Fetch-* headers for modern browser simulation

### Rate Limiting Strategies
- **Conservative**: 1 concurrent download (Greek Language, Cyprus, Panelladikes, Kallipos)
- **Moderate**: 2 concurrent downloads (Kodiko)
- **Original**: 3 concurrent downloads (base implementation)

### Error Handling Enhancements
- **PDF Validation**: Content signature verification
- **URL Fixing**: Automatic correction of malformed URLs
- **Exponential Backoff**: Progressive retry delays
- **Multiple Methods**: GET/POST fallback strategies

## Usage Examples

### Basic Usage Pattern
```bash
python downloader_[site].py \
  --json ../../scraping/json_sitemaps/[site]_pdf.json \
  --type pdf \
  --req get \
  --output ../../downloads/[site] \
  --batch 5 \
  --sleep 3
```

### Automated Download
```bash
bash download_all_[site].sh
```

### Progress Monitoring
```bash
python3 monitor_[site].py
```

## Contributing Guidelines

When working with a new repository:

1. **Start Conservative**: Begin with 1 concurrent download and high sleep times
2. **Analyze Responses**: Study error patterns and server behavior
3. **Site-Specific Customization**: Create specialized version if needed
4. **Document Findings**: Update README with results and modifications
5. **Respectful Scraping**: Always prioritize server resources and terms of service

## Technical Architecture

### Base Components (from original downloader.py)
- **Semaphore control**: Concurrent download limiting
- **User agent rotation**: Bot detection avoidance
- **Progress persistence**: JSON-based progress tracking
- **Error recovery**: Retry mechanisms with delays
- **Flexible parameters**: Command-line configuration

### Specialized Enhancements
- **Site-specific headers**: Tailored for each repository
- **Content validation**: PDF signature verification
- **URL processing**: Malformed URL detection and fixing
- **Advanced logging**: Dual console/file logging
- **Greek language support**: Proper character encoding

## Lessons Learned

### Successful Strategies
1. **Site Analysis**: Understanding each repository's specific requirements
2. **Conservative Approach**: Starting with minimal load and scaling up
3. **Comprehensive Headers**: Modern browser simulation
4. **Error Recovery**: Robust retry mechanisms with exponential backoff
5. **Progress Tracking**: Reliable resumption capabilities

### Common Challenges
1. **Rate Limiting**: Academic repositories often implement strict limits
2. **Bot Detection**: Advanced sites can detect automated access
3. **Server Errors**: Some repositories return 5xx errors regardless of approach
4. **Authentication**: Some sites require login or special sessions
5. **URL Formats**: Malformed or encoded URLs require special handling

## Future Improvements

Potential enhancements for the downloader collection:

1. **Configuration Files**: Site-specific YAML/JSON configuration
2. **Proxy Support**: Rotation to avoid IP-based blocking
3. **Browser Automation**: Selenium/Playwright integration for complex sites
4. **Session Management**: Persistent authentication handling
5. **Metadata Extraction**: Enhanced document information capture
6. **Performance Analytics**: Download statistics and optimization metrics

## Acknowledgments

- **Original Author**: Nikos Tsekos (`downloader.py`)
- **Specializations**: Developed for GlossAPI contribution
- **Framework**: Built on GlossAPI infrastructure by ΕΕΛΛΑΚ
- **Purpose**: Supporting open Greek language model development

---

*This documentation serves as both a technical reference and a contribution guide for the GlossAPI project's document collection efforts.*