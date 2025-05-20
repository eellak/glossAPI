# Downloader Script Contribution

This document details the contribution made to the GlossAPI project, specifically focused on improving the downloader scripts for various repositories.

## Problem Statement

GlossAPI needs to collect texts from various sources to build comprehensive Greek language datasets. However, the default downloader script (`downloader.py`) encountered issues with certain repositories, particularly with the Kallipos repository which consistently returned HTTP 500 errors. Additionally, there was a need for better monitoring of download progress and a more robust approach to downloading from multiple repositories.

## Solution Implemented

1. **Created specialized downloader scripts** for specific repositories:
   - `downloader_kallipos.py` - Customized for the Kallipos repository
   - `downloader_kodiko.py` - Optimized for the Kodiko legal repository

2. **Enhanced download capabilities**:
   - Improved error handling and retry logic
   - Added better browser simulation with comprehensive headers
   - Implemented proper handling of redirects
   - Added SSL verification bypass for problematic sites
   - Customized concurrency and sleep times per repository

3. **Created automation tools**:
   - `download_all_kodiko.sh` - Script to automate the complete download process
   - `monitor_progress.py` - Tool to track download progress in real-time

4. **Comprehensive documentation**:
   - `README_KALLIPOS.md` - Detailed documentation of the Kallipos download attempts
   - `README_KODIKO.md` - Documentation of the successful Kodiko download approach

## Results

### Kallipos Repository
Despite numerous improvements to the downloader script, all attempts to download from Kallipos continued to result in HTTP 500 errors. This suggests server-side restrictions or structural changes to their repository system.

### Kodiko Repository
Successfully downloaded 403+ PDF files from the Kodiko repository, demonstrating the effectiveness of the customized approach. The download process continues automatically with robust error handling.

## Technical Details

### Key Improvements in Downloader Scripts

1. **Enhanced Browser Simulation**
   ```python
   headers = {
       'User-Agent': user_agent,
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
       'Accept-Language': 'en-US,en;q=0.5',
       'Accept-Encoding': 'gzip, deflate, br',
       'Referer': 'https://www.google.com/',
       'Connection': 'keep-alive',
       'Upgrade-Insecure-Requests': '1',
       # Additional headers...
   }
   ```

2. **Improved Error Handling**
   ```python
   try:
       async with requester(pdf_url, headers=headers, allow_redirects=True, max_redirects=5) as response:
           if response.status == 200:
               # Handle successful response
           elif response.status in (301, 302, 303, 307, 308):
               # Handle redirects
           else:
               # Handle errors
   except aiohttp.ClientError as e:
       # Handle client errors
   except asyncio.TimeoutError:
       # Handle timeouts
   except Exception as e:
       # Handle unexpected errors
   ```

3. **Repository-Specific Optimizations**
   - Kallipos: Single concurrent download, longer sleep times, comprehensive headers
   - Kodiko: Moderate concurrency (2), shorter sleep times, targeted referer

### Monitoring Tool Features

The monitoring tool provides real-time statistics on:
- Total entries in the JSON file (total downloads needed)
- Current download count and percentage
- Progress from the progress report JSON file

## Future Improvements

Based on our experience, several improvements could be made to the GlossAPI downloader system:

1. **Modular Architecture**: Create a base downloader class with repository-specific extensions
2. **Configuration-Driven Approach**: Move repository-specific settings to configuration files
3. **Advanced Authentication**: Add support for sites requiring login or session handling
4. **Proxy Support**: Implement proxy rotation to avoid IP-based rate limiting
5. **Browser Automation**: For sites with strict anti-scraping measures, integrate Selenium or Playwright

## Conclusion

This contribution demonstrates a methodical approach to solving a specific problem within the GlossAPI project. By creating specialized tools and documentation, we've improved the project's capability to download from legal repositories like Kodiko, while documenting the challenges with other repositories like Kallipos. The approach follows software engineering best practices of modularization, error handling, and comprehensive documentation.
