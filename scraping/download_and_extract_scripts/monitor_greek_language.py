import os
import json
import time

def get_total_files():
    """Get the total number of files in the JSON sitemap."""
    try:
        with open('../../scraping/json_sitemaps/greek-language_pdf.json', 'r') as f:
            data = json.load(f)
        return len(data)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return 0

def get_downloaded_files():
    """Get the number of downloaded files."""
    try:
        # Count files in directory
        output_dir = '../../downloads/greek-language'
        if not os.path.exists(output_dir):
            return 0
        files = [f for f in os.listdir(output_dir) if f.endswith('.pdf')]
        return len(files)
    except Exception as e:
        print(f"Error counting files: {e}")
        return 0

def get_progress_report():
    """Get information from the progress report."""
    try:
        with open('../../downloads/greek-language/progress_report.json', 'r') as f:
            data = json.load(f)
        return len(data)
    except Exception as e:
        print(f"Error reading progress report: {e}")
        return 0

def main():
    total = get_total_files()
    print(f"Total files to download: {total}")
    
    while True:
        downloaded = get_downloaded_files()
        progress_report = get_progress_report()
        
        if total > 0:
            percent = (downloaded / total) * 100
        else:
            percent = 0
            
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Downloaded: {downloaded}/{total} files ({percent:.2f}%)")
        print(f"Progress from progress_report.json: {progress_report} entries")
        
        time.sleep(10)  # Update every 10 seconds

if __name__ == "__main__":
    main()
