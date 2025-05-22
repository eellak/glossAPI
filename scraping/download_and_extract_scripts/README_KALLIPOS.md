# GlossAPI Downloader - Kallipos Implementation

## Kallipos Downloader Challenges & Solutions

### Initial Problem
When attempting to download PDFs from Kallipos using the default `downloader.py` script in GlossAPI, we encountered consistent HTTP 500 errors (Internal Server Error) from the Kallipos server. The logs showed:

```
2025-05-20 20:39:47,439 - INFO - Arguments received: JSON file: ../../scraping/json_sitemaps/kallipos_pdf.json, Sleep time: 1, File type: pdf, Request type: get, Output path: ../../downloads/kallipos, 'progress_report.json' path: ../../downloads/kallipos
2025-05-20 20:39:47,449 - INFO - Existing progress report found and loaded
2025-05-20 20:39:47,449 - INFO - Starting PDF downloads
2025-05-20 20:39:50,102 - ERROR - Failed to download https://repository.kallipos.gr/retrieve/257938a8-2fba-4151-8100-5c0342d8ff71/295-TRIANTAFYLLOU-Information-Retrieval-and-Search-Techniques.pdf. Status code: 500
```

### Modifications Made
Created a specialized version of the downloader script called `downloader_kallipos.py` with the following improvements:

1. **Ultra-Conservative Approach**
   - Reduced semaphore from 3 to 1 (only one download at a time)
   - This minimizes server load and reduces chances of triggering rate limits

2. **Enhanced Browser Simulation**
   - Added comprehensive HTTP headers to mimic a real browser authentically
   - Updated User-Agent strings to use recent browser versions (90-120)
   - Implemented proper handling of cookies and sessions

3. **Multiple HTTP Method Support**
   - Added fallback to POST requests when GET fails with 500 errors
   - Implemented method iteration to try different approaches

4. **Extended Timeouts & Delays**
   - Increased timeout from 60 to 120 seconds for slower responses
   - Added longer sleep times between requests (up to +5 seconds)
   - Implemented more gradual approach to downloading

5. **Enhanced Redirect Handling**
   - Increased max redirects from 5 to 10
   - Better handling of complex redirect chains

6. **Technical Improvements**
   - Disabled SSL verification for compatibility issues
   - Fixed path handling for progress_report.json
   - Added comprehensive directory creation

### Shell Scripts & Monitoring
Created auxiliary scripts for automation:

1. **download_all_kallipos.sh**: Shell script for automated downloading
2. **monitor_kallipos.py**: Real-time monitoring of download progress

### Command to Run
The modified script should be run with:

```bash
python downloader_kallipos.py --json ../../scraping/json_sitemaps/kallipos_pdf.json --type pdf --req get --output ../../downloads/kallipos --batch 3 --sleep 10
```

Key changes in command parameters:
- Using the specialized `downloader_kallipos.py` script
- Reduced batch size from 10 to 3
- Increased sleep time to 10 seconds for maximum politeness

### Results & Analysis
Despite all the comprehensive improvements made to the downloader script, the Kallipos server continued to respond with HTTP 500 errors. This suggests that the issue is likely:

1. **Server-side restrictions** - Kallipos may have implemented stricter security measures
2. **Authentication requirements** - The repository might now require authentication to access PDFs
3. **API changes** - The PDF URLs or access methods might have been modified
4. **Rate limiting** - Even single requests might be triggering sophisticated bot detection

Common persistent errors:
```
ERROR - Failed to download https://repository.kallipos.gr/retrieve/257938a8-2fba-4151-8100-5c0342d8ff71/295-TRIANTAFYLLOU-Information-Retrieval-and-Search-Techniques.pdf. Status code: 500
ERROR - Failed to download https://repository.kallipos.gr/retrieve/e51ef661-b962-4b35-b170-f4ecd02a3188/562-DASSIOS-Partial-Differential-Equations.pdf. Status code: 500
```

### Technical Specifications

- **Concurrent downloads**: 1 simultaneous connection (ultra-conservative)
- **Request delay**: 1-6 seconds between requests (longest range)
- **Timeout**: 120 seconds per request (double the standard)
- **Retry strategy**: Single retry with method fallback
- **SSL verification**: Disabled for compatibility
- **Multiple HTTP methods**: GET with POST fallback
- **Max redirects**: 10 (highest among all downloaders)
- **User agent rotation**: Modern browser versions (90-120)

### Comparison with Successful Implementations

In contrast, our site-specific approach was highly successful with other repositories:
- **Kodiko**: 86.95% success rate (23,086/26,552 files)
- **Greek Language**: Complete success with specialized approach
- **Cyprus Exams**: Effective with PDF validation
- **Panelladikes**: Good success rate with URL fixing

### Recommendations for Future Work

To successfully download from Kallipos, more advanced techniques might be needed:

1. **Browser automation** (Selenium/Playwright) to handle JavaScript
2. **Session management** with proper authentication flow
3. **Proxy rotation** to avoid IP-based blocking
4. **Manual analysis** of the Kallipos access mechanism
5. **Contact with Kallipos** administrators for API access

The Kallipos case demonstrates that some academic repositories have sophisticated protection mechanisms that require specialized approaches beyond traditional web scraping techniques.