# GlossAPI Downloader for Kodiko

## Kodiko Downloader Implementation

### Background
After our unsuccessful attempts with Kallipos, we've created a specialized version of the downloader for the Kodiko website, which hosts legal texts and documents in PDF format.

### Modifications Made
Created `downloader_kodiko.py` with the following customizations:

1. **Moderate Concurrency**
   - Using a semaphore value of 2 (instead of 1 or 3)
   - This balances speed with server load considerations

2. **Browser Simulation**
   - Added comprehensive HTTP headers to mimic a real browser
   - Using Google as a referer which is common for legal document searches
   - Keeping cookies and session data between requests

3. **Error Handling**
   - Implemented proper redirects following (up to 5 redirects)
   - Improved error reporting and handling

4. **Rate Limiting**
   - Using moderate sleep times between requests (basic value + random 0-3 seconds)
   - This avoids overwhelming the server while maintaining reasonable download speed

5. **Technical Improvements**
   - Disabled SSL verification to avoid certificate issues
   - Added proper output directory handling
   - Enhanced logging for better troubleshooting

### Command to Run
Execute the following command to run the Kodiko downloader:

```bash
python downloader_kodiko.py --json ../../scraping/json_sitemaps/kodiko_pdf.json --type pdf --req get --output ../../downloads/kodiko --batch 5 --sleep 3
```

Key parameters:
- Using specialized `downloader_kodiko.py` script
- Moderate batch size of 5
- Sleep time of 3 seconds between requests
- GET request method which should work fine for Kodiko
- Separate output directory for Kodiko documents

### Results
The Kodiko downloader was successful! Here's the output from our first run:

```
2025-05-20 20:53:04,554 - INFO - Arguments received: JSON file: ../../scraping/json_sitemaps/kodiko_pdf.json, Sleep time: 3, File type: pdf, Request type: get, Output path: ../../downloads/kodiko, 'progress_report.json' path: ../../downloads/kodiko
2025-05-20 20:53:04,631 - INFO - No existing progress report found
2025-05-20 20:53:04,631 - INFO - Starting PDF downloads
2025-05-20 20:53:08,954 - INFO - Downloaded paper_2.pdf
2025-05-20 20:53:11,673 - INFO - Downloaded paper_1.pdf
2025-05-20 20:53:14,526 - INFO - Downloaded paper_3.pdf
2025-05-20 20:53:18,134 - INFO - Downloaded paper_4.pdf
2025-05-20 20:53:23,776 - INFO - Downloaded paper_5.pdf
2025-05-20 20:53:23,778 - INFO - PDF downloads completed
2025-05-20 20:53:23,778 - INFO - Progress report written to progress_report.json
```

All 5 PDFs in the batch were successfully downloaded, without any errors. This confirms that our approach with Kodiko is working correctly.

## Comparison with Kallipos

The Kodiko site appears to have a different structure than Kallipos:
- It hosts legal documents that are generally public and should be more accessible
- The site may have less strict anti-scraping measures
- The PDFs are likely served directly rather than through a complex retrieval system

This implementation strikes a balance between being respectful of the server's resources while attempting to efficiently download the documents.
