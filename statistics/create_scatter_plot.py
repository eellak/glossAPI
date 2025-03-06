import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Read the CSV file
df = pd.read_csv('../full_clean_output/statistics.csv')

# Filter the data according to specifications
filtered_df = df[
    (df['Total Paragraphs'] <= 3000) & 
    (df['ΠΕΡΙΕΧΟΜΕΝΑ'] <= 400)
]

# Set the style for better visualization
plt.style.use('seaborn')
plt.figure(figsize=(16, 10))

# Create scatter plot with alpha transparency and small point size
# Using hexbin for better visualization of dense areas
plt.hexbin(filtered_df['Total Paragraphs'], 
           filtered_df['ΠΕΡΙΕΧΟΜΕΝΑ'],
           gridsize=50,  # Adjust grid size for hexagonal binning
           cmap='YlOrRd',  # Color map that works well for density
           mincnt=1,  # Minimum number of points for a hex to be colored
           bins='log')  # Logarithmic binning for better color distribution

# Add a colorbar
plt.colorbar(label='Number of documents (log scale)')

# Customize the plot
plt.xlabel('Total Paragraphs', fontsize=12)
plt.ylabel('ΠΕΡΙΕΧΟΜΕΝΑ', fontsize=12)
plt.title('Document Statistics: Paragraphs vs ΠΕΡΙΕΧΟΜΕΝΑ\n(Density Visualization)', fontsize=14)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot with high DPI for better quality
plt.savefig('plots/scatter_plot.png', dpi=300, bbox_inches='tight')
