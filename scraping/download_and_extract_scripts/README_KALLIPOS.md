# GlossAPI Downloader Observations & Modifications

## Kallipos Downloader Issues

### Initial Problem
When attempting to download PDFs from Kallipos using the default `downloader.py` script in GlossAPI, we encountered consistent HTTP 500 errors (Internal Server Error) from the Kallipos server. The logs showed:

```
2025-05-20 20:39:47,439 - INFO - Arguments received: JSON file: ../../scraping/json_sitemaps/kallipos_pdf.json, Sleep time: 1, File type: pdf, Request type: get, Output path: ../../downloads/kallipos, 'progress_report.json' path: ../../downloads/kallipos
2025-05-20 20:39:47,449 - INFO - Existing progress report found and loaded
2025-05-20 20:39:47,449 - INFO - Starting PDF downloads
2025-05-20 20:39:50,102 - ERROR - Failed to download https://repository.kallipos.gr/retrieve/257938a8-2fba-4151-8100-5c0342d8ff71/295-TRIANTAFYLLOU-Information-Retrieval-and-Search-Techniques.pdf. Status code: 500
```

The command used:
```bash
python downloader.py --json ../../scraping/json_sitemaps/kallipos_pdf.json --type pdf --req get --output ../../downloads/kallipos --batch 10
```

### Modifications Made
Created a specialized version of the downloader script called `downloader_kallipos.py` with the following improvements:

1. **Reduced Concurrency**
   - Changed the semaphore from 3 to 1 to limit to only one download at a time
   - This reduces server load and minimizes chances of triggering rate limits

2. **Enhanced Browser Simulation**
   - Added more HTTP headers to mimic a real browser more authentically
   - Updated User-Agent strings to use more recent browser versions
   - Implemented proper handling of cookies and sessions

3. **Improved Error Handling**
   - Added fallback to POST requests when GET fails
   - Implemented proper following of redirects
   - Improved timeout and error recovery strategies

4. **Rate Limiting & Politeness**
   - Increased sleep time between requests
   - Added randomized delays to avoid detection
   - Implemented more gradual approach to downloading

5. **Technical Improvements**
   - Disabled SSL verification that was causing issues
   - Fixed path handling for progress_report.json
   - Added directory creation if output directory doesn't exist

### New Command to Run
The modified script should be run with:

```bash
python downloader_kallipos.py --json ../../scraping/json_sitemaps/kallipos_pdf.json --type pdf --req get --output ../../downloads/kallipos --batch 3 --sleep 10
```

Key changes in command parameters:
- Using the specialized `downloader_kallipos.py` script
- Reduced batch size from 10 to 3
- Increased sleep time from 1 to 10 seconds

### Results
Despite all the improvements made to the downloader script, there was no change in the results. The Kallipos server continued to respond with HTTP 500 errors. This suggests that the issue might be:

1. On the server side - Kallipos might have implemented stricter security measures or their API might have changed
2. Authentication requirements - The repository might now require authentication to access PDFs
3. Structural changes - The PDF URLs or access methods might have been modified

Common errors encountered:
```
2025-05-20 20:39:50,102 - ERROR - Failed to download https://repository.kallipos.gr/retrieve/257938a8-2fba-4151-8100-5c0342d8ff71/295-TRIANTAFYLLOU-Information-Retrieval-and-Search-Techniques.pdf. Status code: 500
2025-05-20 20:39:50,117 - ERROR - Failed to download https://repository.kallipos.gr/retrieve/e51ef661-b962-4b35-b170-f4ecd02a3188/562-DASSIOS-Partial-Differential-Equations.pdf. Status code: 500
```

In contrast, our approach was successful with the Kodiko repository, where we managed to download 403 PDF files successfully using our modified downloader script.

Further investigation with more advanced techniques (like browser automation) might be needed to successfully download from Kallipos.

## Other Sites
[Space for documenting issues and solutions for other sites]

## General Observations about GlossAPI Downloader

The GlossAPI downloader script provides a flexible way to download documents from various sources, but requires site-specific customization in many cases. Common challenges include:

1. **Rate Limiting**: Many academic repositories implement rate limiting to prevent excessive downloading
2. **Bot Detection**: Advanced websites can detect and block automated downloading
3. **Authentication**: Some repositories require login or special sessions
4. **Server-Side Issues**: Sometimes servers respond with 5xx errors due to internal issues

For contributing effectively to GlossAPI, it's important to customize downloaders for each specific repository and implement polite scraping practices.

## Future Improvements

Potential improvements to make the downloader more robust:

1. Site-specific configuration files rather than hard-coded parameters
2. Proxy rotation to avoid IP-based blocking
3. Session persistence between runs
4. Automatic retries with exponential backoff
5. Better metadata handling and extraction
6. More sophisticated browser simulation (e.g. headless browser integration)

## Contributing Guidelines

When working on downloading from a new site:

1. Start with small batches and long sleep times
2. Analyze response patterns and errors
3. Create a site-specific version of the downloader if needed
4. Document your findings and modifications
5. Be respectful of the target website's resources