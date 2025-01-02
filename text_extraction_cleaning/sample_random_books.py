import pandas as pd
import numpy as np

# Read the parquet file
df = pd.read_parquet('paragraphs_output.parquet')

# Get list of unique books (source_files)
unique_books = df['source_file'].unique()

# Sample 3 random books
random_books = np.random.choice(unique_books, size=3, replace=False)

# For each random book, create a CSV file with its paragraphs in order
for book in random_books:
    # Filter paragraphs for this book
    book_df = df[df['source_file'] == book].copy()
    
    # Save to CSV with the same filename but .csv extension
    output_filename = book.replace('.md', '.csv')
    book_df.to_csv(output_filename, index=False)
    print(f"Created {output_filename} with {len(book_df)} paragraphs")
