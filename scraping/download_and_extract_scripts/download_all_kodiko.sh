#!/bin/bash

# Set variables
JSON_FILE="../../scraping/json_sitemaps/kodiko_pdf.json"
OUTPUT_DIR="../../downloads/kodiko"
BATCH_SIZE=10  # Increased batch size for faster downloads
SLEEP_TIME=2   # Slightly reduced sleep time for faster downloads
MAX_ITERATIONS=1000  # Safety limit to prevent infinite loops

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Initialize counter
iteration=0

# Run the downloader in a loop until all files are downloaded
while [ $iteration -lt $MAX_ITERATIONS ]; do
    echo "Running iteration $((iteration+1))..."
    
    # Run the downloader
    python downloader_kodiko.py --json $JSON_FILE --type pdf --req get --output $OUTPUT_DIR --batch $BATCH_SIZE --sleep $SLEEP_TIME
    
    # Wait a moment for the logs to flush
    sleep 1
    
    # Get the current file count
    before_count=$(ls -1 "$OUTPUT_DIR"/*.pdf 2>/dev/null | wc -l)
    
    # If no new files were downloaded in this iteration, we're probably done
    if [ $iteration -gt 0 ] && [ $before_count -eq $after_count ]; then
        echo "No new files downloaded in this iteration. Exiting..."
        break
    fi
    
    # Store the count for the next iteration
    after_count=$before_count
    
    # Increment counter
    iteration=$((iteration+1))
    
    # Small pause between iterations to let the system breathe
    sleep 3
done

# Count the number of downloaded PDFs
pdf_count=$(ls -1 $OUTPUT_DIR/*.pdf 2>/dev/null | wc -l)
echo "Total number of downloaded PDFs: $pdf_count"

# Create a summary file
echo "Download Summary" > $OUTPUT_DIR/download_summary.txt
echo "----------------" >> $OUTPUT_DIR/download_summary.txt
echo "Date: $(date)" >> $OUTPUT_DIR/download_summary.txt
echo "Total PDFs downloaded: $pdf_count" >> $OUTPUT_DIR/download_summary.txt
echo "JSON file used: $JSON_FILE" >> $OUTPUT_DIR/download_summary.txt
echo "Batch size: $BATCH_SIZE" >> $OUTPUT_DIR/download_summary.txt
echo "Sleep time: $SLEEP_TIME seconds" >> $OUTPUT_DIR/download_summary.txt
echo "Total iterations: $iteration" >> $OUTPUT_DIR/download_summary.txt

echo "Download complete! Summary saved to $OUTPUT_DIR/download_summary.txt"
