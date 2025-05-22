# GlossAPI Downloader for Kodiko

## Kodiko Downloader Implementation

### Background
This specialized downloader was developed for the Kodiko website (kodiko.gr), which hosts legal texts and documents in PDF format. After successful testing, we achieved excellent download rates.

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

### Results & Performance
The Kodiko downloader achieved excellent results:

**Download Statistics:**
- **Total files**: 26,552
- **Successfully downloaded**: 23,086 files 
- **Success rate**: 86.95%
- **Status**: Download completed on 2025-05-22

```
[2025-05-22 14:49:02] Downloaded: 23078/26552 files (86.92%)
Progress from progress_report.json: 23078 entries
[2025-05-22 14:49:12] Downloaded: 23081/26552 files (86.93%)
Progress from progress_report.json: 23078 entries
[2025-05-22 14:49:22] Downloaded: 23086/26552 files (86.95%)
Progress from progress_report.json: 23078 entries
```

The high success rate demonstrates that our site-specific approach was effective for the Kodiko repository.

### Shell Scripts & Monitoring
Created auxiliary scripts for automation:

1. **download_all_kodiko.sh**: Shell script for automated downloading
2. **monitor_kodiko.py**: Real-time monitoring of download progress

## Comparison with Other Sites

The Kodiko site characteristics:
- Hosts legal documents that are generally public and accessible
- Less strict anti-scraping measures compared to academic repositories
- PDFs are served directly rather than through complex retrieval systems
- Good server response times and stability

This implementation strikes a balance between being respectful of the server's resources while efficiently downloading the legal document collection.

## Technical Specifications

- **Concurrent downloads**: 2 simultaneous connections
- **Request delay**: 1-4 seconds between requests  
- **Timeout**: 60 seconds per request
- **Retry strategy**: Single retry on failure
- **SSL verification**: Disabled for compatibility
- **Default protocol**: HTTPS with HTTP fallback
- **User agent rotation**: Modern browser versions (90-120)

## Contributing to GlossAPI

This implementation serves as a good template for other legal document repositories. The balanced approach of moderate concurrency with respectful delays proved successful for large-scale document collection.