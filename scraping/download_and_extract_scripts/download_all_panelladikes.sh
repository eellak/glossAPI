#!/bin/bash

# This script downloads all Panhellenic Exams PDF files
echo "Starting Panhellenic Exams PDF downloads with fixed script..."

# Create output directory if it doesn't exist
mkdir -p ../../downloads/panelladikes-exams

# Create progress directory if it doesn't exist
mkdir -p progress_reports

# Create logs directory if it doesn't exist
mkdir -p logs

# Run iterations with different batch sizes
echo "Running iteration 1..."
python3 downloader_panelladikes_fixed.py --json ../../scraping/json_sitemaps/themata-lyseis-panelladikwn_pdf.json --type pdf --req get --output ../../downloads/panelladikes-exams --batch 5 --sleep 3 --little_potato progress_reports

echo "Download complete!"
