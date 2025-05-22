#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Monitor script for Panhellenic (Panelladikes) Exams PDFs download progress.
This script monitors the progress of the download process for Panhellenic Exams PDFs.
"""

import os
import json
import argparse
from collections import Counter
from datetime import datetime

def monitor_progress(progress_dir, json_file=None):
    """
    Monitors the progress of the download process
    
    Args:
        progress_dir (str): Directory containing progress files
        json_file (str, optional): Path to the JSON file with the source data
    """
    # Find the progress file
    progress_file = os.path.join(progress_dir, 'progress_report.json')
    
    if not os.path.exists(progress_file):
        print(f"Progress file not found: {progress_file}")
        return
    
    # Load progress
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress = json.load(f)
    except json.JSONDecodeError:
        print(f"Could not parse progress file {progress_file}")
        return
    
    # Count by status
    status_counts = Counter(item["status"] for item in progress)
    
    # Calculate percentages
    total = len(progress)
    
    # Load source data if available
    source_total = None
    if json_file and os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                source_data = json.load(f)
                if isinstance(source_data, dict):
                    source_total = len(source_data)
                elif isinstance(source_data, list):
                    source_total = len(source_data)
        except json.JSONDecodeError:
            print(f"Could not parse source JSON file {json_file}")
    
    # Print header
    print("\n" + "="*80)
    print(f" PANHELLENIC EXAMS DOWNLOAD PROGRESS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Print statistics
    print(f"\nTotal processed: {total}")
    if source_total:
        percent_of_total = (total / source_total) * 100
        print(f"Out of {source_total} total items ({percent_of_total:.1f}%)")
    
    print("\nStatus breakdown:")
    for status, count in status_counts.items():
        percent = (count / total) * 100 if total > 0 else 0
        print(f"  {status}: {count} ({percent:.1f}%)")
    
    # Print recent items
    print("\nMost recent downloads:")
    if progress:
        for item in progress[-10:]:
            status_emoji = "✅" if item["status"] == "success" else "⚠️" if item["status"] == "already_exists" else "❌"
            print(f"  {status_emoji} {item['title'][:50]}{'...' if len(item['title']) > 50 else ''}")
    
    # Print footer
    print("\n" + "="*80)


def main():
    """Main function to handle command line arguments and start the monitoring process"""
    parser = argparse.ArgumentParser(description='Monitor Panhellenic Exams PDFs download progress')
    parser.add_argument('--progress', default='progress_reports', help='Directory containing progress files')
    parser.add_argument('--json', default='../../scraping/json_sitemaps/themata-lyseis-panelladikwn_pdf.json', 
                        help='Path to the JSON file with the source data')
    
    args = parser.parse_args()
    
    monitor_progress(args.progress, args.json)


if __name__ == "__main__":
    main()
