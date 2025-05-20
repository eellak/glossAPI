#!/usr/bin/env python3
"""
Monitor progress of downloads by checking the progress_report.json file
and the number of files downloaded.
"""

import os
import json
import argparse
import time
from datetime import datetime

def count_files(directory, extension='.pdf'):
    """Count the number of files with a specific extension in a directory"""
    try:
        return len([f for f in os.listdir(directory) if f.endswith(extension)])
    except FileNotFoundError:
        print(f"Directory {directory} not found")
        return 0

def get_total_entries(json_file):
    """Get the total number of entries in the JSON file"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            return len(data)
    except FileNotFoundError:
        print(f"JSON file {json_file} not found")
        return 0
    except json.JSONDecodeError:
        print(f"Error parsing JSON file {json_file}")
        return 0

def get_progress(progress_file):
    """Get the progress from the progress_report.json file"""
    try:
        with open(progress_file, 'r') as f:
            data = json.load(f)
            return len(data)
    except FileNotFoundError:
        print(f"Progress file {progress_file} not found")
        return 0
    except json.JSONDecodeError:
        print(f"Error parsing progress file {progress_file}")
        return 0

def main():
    parser = argparse.ArgumentParser(description='Monitor download progress')
    parser.add_argument('--json', help='Path to JSON file with URLs', required=True)
    parser.add_argument('--output', help='Directory where files are downloaded', required=True)
    parser.add_argument('--interval', type=int, default=10, help='Refresh interval in seconds')
    parser.add_argument('--continuous', action='store_true', help='Run continuously until all files are downloaded')
    
    args = parser.parse_args()
    
    json_file = args.json
    output_dir = args.output
    progress_file = os.path.join(output_dir, 'progress_report.json')
    interval = args.interval
    continuous = args.continuous
    
    total_entries = get_total_entries(json_file)
    print(f"Total entries in JSON file: {total_entries}")
    
    try:
        while True:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file_count = count_files(output_dir)
            progress_count = get_progress(progress_file)
            
            progress_percent = (file_count / total_entries) * 100 if total_entries > 0 else 0
            
            print(f"[{current_time}] Downloaded: {file_count}/{total_entries} files ({progress_percent:.2f}%)")
            print(f"Progress from progress_report.json: {progress_count} entries")
            
            if file_count >= total_entries and not continuous:
                print("All files downloaded!")
                break
                
            if not continuous:
                break
                
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
        
if __name__ == "__main__":
    main()
