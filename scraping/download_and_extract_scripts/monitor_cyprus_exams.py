#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
from datetime import datetime

def monitor_progress(json_file_path, output_dir, interval=5):
    """
    Monitor the progress of the cyprus-exams downloader.
    
    Args:
        json_file_path: Path to the JSON file with sitemap.
        output_dir: Output directory where files are downloaded.
        interval: Interval in seconds to check progress.
    """
    try:
        # Load the JSON file to get the total number of items
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Count PDF links only
        total_files = sum(1 for item in data if 'link' in item and 'pdf' in item['link'].lower())
        
        progress_report_path = os.path.join(output_dir, 'progress_report.json')
        
        while True:
            try:
                # Count downloaded files
                downloaded_files = len([f for f in os.listdir(output_dir) if f.endswith('.pdf')])
                
                # Get progress from progress_report.json if it exists
                progress_from_report = 0
                if os.path.exists(progress_report_path):
                    with open(progress_report_path, 'r', encoding='utf-8') as f:
                        progress_report = json.load(f)
                        progress_from_report = len(progress_report)
                
                # Calculate percentage
                percentage = (downloaded_files / total_files) * 100 if total_files > 0 else 0
                
                # Print progress
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Downloaded: {downloaded_files}/{total_files} files ({percentage:.2f}%)")
                print(f"Progress from progress_report.json: {progress_from_report} entries")
                
                # Wait for the next check
                time.sleep(interval)
                
            except Exception as e:
                print(f"Error checking progress: {e}")
                time.sleep(interval)
    
    except Exception as e:
        print(f"Error initializing progress monitor: {e}")

if __name__ == "__main__":
    json_file_path = "../../scraping/json_sitemaps/cyprus-exams_pdf.json"
    output_dir = "../../downloads/cyprus-exams"
    
    print(f"Monitoring progress for {json_file_path}...")
    monitor_progress(json_file_path, output_dir)
