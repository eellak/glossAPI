import pandas as pd
import numpy as np
import os
import shutil

# Read the CSV file
df = pd.read_csv('../full_clean_output/statistics.csv')

# Filter the data according to original specifications
filtered_df = df[
    (df['Total Paragraphs'] <= 3000) & 
    (df['ΠΕΡΙΕΧΟΜΕΝΑ'] <= 400)
]

# Define the groups
group1 = filtered_df[filtered_df['ΠΕΡΙΕΧΟΜΕΝΑ'] <= 25]  # Low ΠΕΡΙΕΧΟΜΕΝΑ group
group2 = filtered_df[filtered_df['ΠΕΡΙΕΧΟΜΕΝΑ'] >= 50]  # High ΠΕΡΙΕΧΟΜΕΝΑ group

# Sample 100 from each group
np.random.seed(42)  # For reproducibility
sample_low = group1.sample(n=100)
sample_high = group2.sample(n=100)

# Create directories
desktop_path = os.path.expanduser('~/Desktop')
low_dir = os.path.join(desktop_path, 'low_periexomena_sample')
high_dir = os.path.join(desktop_path, 'high_periexomena_sample')

os.makedirs(low_dir, exist_ok=True)
os.makedirs(high_dir, exist_ok=True)

# Function to copy files and create info file
def copy_sample_files(sample_df, target_dir, group_name):
    # Create info file
    info_path = os.path.join(target_dir, 'sample_info.csv')
    sample_df.to_csv(info_path, index=False)
    
    # Copy files
    source_dir = '../full_clean_output'
    for filename in sample_df['filename']:
        clean_filename = f"clean_{filename}"  # Add 'clean_' prefix
        src = os.path.join(source_dir, clean_filename)
        dst = os.path.join(target_dir, clean_filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"Warning: Source file not found: {src}")
    
    # Print summary
    print(f"\n{group_name} Sample Summary:")
    print(f"Files copied to: {target_dir}")
    print(f"Total Paragraphs - Mean: {sample_df['Total Paragraphs'].mean():.1f}, Median: {sample_df['Total Paragraphs'].median():.1f}")
    print(f"ΠΕΡΙΕΧΟΜΕΝΑ - Mean: {sample_df['ΠΕΡΙΕΧΟΜΕΝΑ'].mean():.1f}, Median: {sample_df['ΠΕΡΙΕΧΟΜΕΝΑ'].median():.1f}")

# Copy files for both groups
copy_sample_files(sample_low, low_dir, "Low ΠΕΡΙΕΧΟΜΕΝΑ")
copy_sample_files(sample_high, high_dir, "High ΠΕΡΙΕΧΟΜΕΝΑ")
