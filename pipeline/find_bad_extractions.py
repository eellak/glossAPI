import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import shutil
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json

# Directories
input_folders = [
    '/mnt/data/raw_md_files/greek_processed_output',
    '/mnt/data/raw_md_files/ypologistis_mou_output'
]
output_folder = '/mnt/data/raw_md_files/cluster_analysis'
os.makedirs(output_folder, exist_ok=True)

def clean_text(text):
    """Remove sequences of dots, dashes, pipes, /gX patterns, and underscores"""
    # Remove sequences of dots, dashes, and pipes with or without spaces
    text = re.sub(r'[\s]*\.{2,}[\s]*', ' ', text)  # Remove ...
    text = re.sub(r'[\s]*\|[\s]*', ' ', text)      # Remove |
    text = re.sub(r'[\s]*-{2,}[\s]*', ' ', text)   # Remove --
    
    ## Remove /gX patterns (where X is any number of digits)
    #text = re.sub(r'/g\d+', '', text)
    
    # Remove sequences of underscores (with optional backslashes)
    text = re.sub(r'[\\]*_+[\\]*', ' ', text)
    
    # Clean up extra spaces that might be left
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_text(text):
    # First clean special characters
    text = clean_text(text)
    # Remove image tags
    text = re.sub(r'<!--\s*image\s*-->', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove markdown headers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    return text.strip()

def process_file(args):
    """Process a single file for parallel execution"""
    filepath, folder = args
    try:
        with open(filepath, 'r', encoding='utf-8') as infile:
            text = infile.read()
        text = preprocess_text(text)
        return (filepath, text, folder)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def get_all_files():
    """Get list of all files to process"""
    all_files = []
    for folder in input_folders:
        for filename in os.listdir(folder):
            if filename.endswith('.md'):
                filepath = os.path.join(folder, filename)
                all_files.append((filepath, folder))
    return all_files

def custom_tokenizer(text):
    """Custom tokenizer to exclude trigrams with only dots, dashes, pipes, or any spaces"""
    trigrams = []
    for i in range(len(text) - 2):
        trigram = text[i:i+3]
        # Skip trigrams that contain spaces or contain only dots, dashes, pipes
        if ' ' not in trigram and not re.match(r'^[\.\-\|]+$', trigram):
            trigrams.append(trigram)
    return trigrams

def main():
    print("Starting document analysis...")
    
    # Get all files
    all_files = get_all_files()
    print(f"Found {len(all_files)} files to process")
    
    # Process files in parallel
    n_cores = max(1, cpu_count() - 1)  # Leave one core free
    print(f"Using {n_cores} CPU cores for parallel processing")
    
    documents = []
    filenames = []
    folder_sources = []
    
    with Pool(n_cores) as pool:
        # Use tqdm to show progress
        results = list(tqdm(
            pool.imap(process_file, all_files),
            total=len(all_files),
            desc="Processing files"
        ))
    
    # Filter out None results and unpack
    for result in results:
        if result is not None:
            filepath, text, folder = result
            documents.append(text)
            filenames.append(os.path.basename(filepath))
            folder_sources.append(folder)
    
    print(f"\nSuccessfully processed {len(documents)} documents")
    
    print("\nCreating trigram representation...")
    vectorizer = TfidfVectorizer(
        analyzer=custom_tokenizer,  # Use custom tokenizer
        lowercase=False,
        max_features=10000  # Limit features for speed
    )
    
    # Use tqdm for vectorization progress
    X = vectorizer.fit_transform(tqdm(documents, desc="Vectorizing documents", unit="doc"))
    print(f"Extracted trigram feature matrix of shape: {X.shape}")
    
    # Cluster the documents
    print("\nPerforming clustering...")
    n_clusters = 2  # Changed to 2 clusters
    kmeans = KMeans(n_clusters=n_clusters, 
                    random_state=42,
                    n_init=10)  # Reduce number of initializations for speed
    
    # Use tqdm for clustering progress
    labels = list(tqdm(
        kmeans.fit_predict(X),
        desc="Clustering documents",
        total=len(documents),
        unit="doc"
    ))
    
    # Calculate silhouette score
    print("\nCalculating Silhouette Score...")
    # Use tqdm to show progress for silhouette score calculation
    score = silhouette_score(
        tqdm(X, desc="Computing Silhouette Score", total=X.shape[0]), 
        labels
    )
    print(f"Silhouette Score for {n_clusters} clusters: {score:.3f}")
    
    # Visualize clusters
    print("\nCreating visualization...")
    # For large datasets, use random sampling for visualization
    sample_size = min(5000, X.shape[0])  # Sample up to 5000 points
    sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
    
    # Perform PCA on the sampled data
    pca = PCA(n_components=2)
    X_dense = X.toarray()
    X_sampled = X_dense[sample_indices]
    labels_sampled = labels[sample_indices]
    
    X_2d = pca.fit_transform(X_sampled)
    
    plt.figure(figsize=(16, 10))
    colors = ['blue', 'red']
    for i in range(n_clusters):
        mask = labels_sampled == i
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], c=colors[i], 
                    label=f'Cluster {i}', alpha=0.6, s=10)  # Reduce point size
    
    plt.title(f'Document Clusters Visualization (PCA, {sample_size} sampled points)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'clusters_visualization.png'), dpi=300)
    plt.close()
    
    # Analyze top trigrams for each cluster
    print("\nAnalyzing top trigrams per cluster...")
    feature_names = vectorizer.get_feature_names_out()
    
    # Dictionary to store top trigrams for JSON
    clusters_top_trigrams = {}
    
    for cluster_idx in range(n_clusters):
        centroid = kmeans.cluster_centers_[cluster_idx]
        top_indices = np.argsort(centroid)[::-1][:50]  # Get top 50
        top_trigrams = [feature_names[i] for i in top_indices]
        
        # Store in dictionary for JSON - just the ordered list
        clusters_top_trigrams[f"cluster_{cluster_idx}"] = top_trigrams
        
        # Print top 15 for console output
        print(f"\nCluster {cluster_idx} top 15 trigrams:")
        print(", ".join(top_trigrams[:15]))
    
    # Save to JSON file
    json_path = os.path.join(output_folder, 'top_trigrams.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(clusters_top_trigrams, f, ensure_ascii=False, indent=2)
    print(f"\nSaved top 50 trigrams for each cluster to {json_path}")
    
    # Copy files to good/bad directories
    print("\nCopying files to good/bad directories...")
    classified_dir = os.path.join(output_folder, 'classified_files')
    good_dir = os.path.join(classified_dir, 'good')
    bad_dir = os.path.join(classified_dir, 'bad')
    
    # Create directories
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(bad_dir, exist_ok=True)
    
    # Based on the trigrams analysis:
    # Cluster 0 (smaller) contains OCR errors -> bad
    # Cluster 1 (larger) contains good text -> good
    
    copied_count = {'good': 0, 'bad': 0}
    
    for filename, label, source_folder in zip(filenames, labels, folder_sources):
        source_path = os.path.join(source_folder, filename)
        if label == 0:  # Bad cluster
            dest_dir = bad_dir
            copied_count['bad'] += 1
        else:  # Good cluster
            dest_dir = good_dir
            copied_count['good'] += 1
            
        dest_path = os.path.join(dest_dir, filename)
        try:
            shutil.copy2(source_path, dest_path)
        except Exception as e:
            print(f"Error copying {filename}: {e}")
    
    print(f"\nFiles copied:")
    print(f"Good files: {copied_count['good']}")
    print(f"Bad files: {copied_count['bad']}")
    
    print("\nAnalysis complete! Check the visualization and classified files in the cluster_analysis folder.")

if __name__ == "__main__":
    main()
