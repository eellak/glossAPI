#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p ../../downloads/cyprus-exams

# Set initial values
ITERATION=1
MAX_ITERATIONS=20
BATCH_SIZE=5
SLEEP_TIME=3
LAST_COUNT=0
CURRENT_COUNT=0

echo "Starting Cyprus Exams PDF downloads..."

# Run the downloader in a loop
while [ $ITERATION -le $MAX_ITERATIONS ]; do
    echo "Running iteration $ITERATION..."
    
    # Run the downloader
    python3 downloader_cyprus_exams.py \
        --json ../../scraping/json_sitemaps/cyprus-exams_pdf.json \
        --type pdf \
        --req get \
        --output ../../downloads/cyprus-exams \
        --batch $BATCH_SIZE \
        --sleep $SLEEP_TIME
    
    # Count the number of downloaded PDFs
    CURRENT_COUNT=$(ls -1 ../../downloads/cyprus-exams/*.pdf 2>/dev/null | wc -l)
    
    # Check if we've downloaded new files in this iteration
    if [ $CURRENT_COUNT -eq $LAST_COUNT ] && [ $CURRENT_COUNT -ne 0 ]; then
        echo "No new files downloaded in this iteration. Exiting..."
        break
    fi
    
    LAST_COUNT=$CURRENT_COUNT
    ITERATION=$((ITERATION + 1))
    
    # Sleep between iterations
    sleep 6
done

# Generate a summary
echo "Total number of downloaded PDFs: $CURRENT_COUNT" | tee ../../downloads/cyprus-exams/download_summary.txt
echo "Download complete! Summary saved to ../../downloads/cyprus-exams/download_summary.txt"
