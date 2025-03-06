import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
middle_group = filtered_df[(filtered_df['ΠΕΡΙΕΧΟΜΕΝΑ'] > 25) & (filtered_df['ΠΕΡΙΕΧΟΜΕΝΑ'] < 50)]

# Calculate statistics for each group
def analyze_group(group, name):
    stats = {
        'Count': len(group),
        'Percentage': len(group) / len(filtered_df) * 100,
        'Total Paragraphs Mean': group['Total Paragraphs'].mean(),
        'Total Paragraphs Median': group['Total Paragraphs'].median(),
        'ΠΕΡΙΕΧΟΜΕΝΑ Mean': group['ΠΕΡΙΕΧΟΜΕΝΑ'].mean(),
        'ΠΕΡΙΕΧΟΜΕΝΑ Median': group['ΠΕΡΙΕΧΟΜΕΝΑ'].median(),
        'Ratio Mean': (group['ΠΕΡΙΕΧΟΜΕΝΑ'] / group['Total Paragraphs']).mean(),
        'Ratio Median': (group['ΠΕΡΙΕΧΟΜΕΝΑ'] / group['Total Paragraphs']).median()
    }
    print(f"\n{name} Group Analysis:")
    print(f"Number of documents: {stats['Count']} ({stats['Percentage']:.1f}% of total)")
    print(f"Total Paragraphs - Mean: {stats['Total Paragraphs Mean']:.1f}, Median: {stats['Total Paragraphs Median']:.1f}")
    print(f"ΠΕΡΙΕΧΟΜΕΝΑ - Mean: {stats['ΠΕΡΙΕΧΟΜΕΝΑ Mean']:.1f}, Median: {stats['ΠΕΡΙΕΧΟΜΕΝΑ Median']:.1f}")
    print(f"ΠΕΡΙΕΧΟΜΕΝΑ/Total Ratio - Mean: {stats['Ratio Mean']:.3f}, Median: {stats['Ratio Median']:.3f}")
    return stats

# Analyze each group
print(f"Total documents in analysis: {len(filtered_df)}")
stats_group1 = analyze_group(group1, "Low ΠΕΡΙΕΧΟΜΕΝΑ (≤25)")
stats_middle = analyze_group(middle_group, "Middle Group (25-50)")
stats_group2 = analyze_group(group2, "High ΠΕΡΙΕΧΟΜΕΝΑ (≥50)")

# Create a scatter plot showing the groups in different colors
plt.figure(figsize=(16, 10))
plt.scatter(group1['Total Paragraphs'], group1['ΠΕΡΙΕΧΟΜΕΝΑ'], 
           alpha=0.5, label='Low ΠΕΡΙΕΧΟΜΕΝΑ (≤25)', color='blue')
plt.scatter(group2['Total Paragraphs'], group2['ΠΕΡΙΕΧΟΜΕΝΑ'], 
           alpha=0.5, label='High ΠΕΡΙΕΧΟΜΕΝΑ (≥50)', color='red')
plt.scatter(middle_group['Total Paragraphs'], middle_group['ΠΕΡΙΕΧΟΜΕΝΑ'], 
           alpha=0.5, label='Middle Group (25-50)', color='green')

plt.xlabel('Total Paragraphs')
plt.ylabel('ΠΕΡΙΕΧΟΜΕΝΑ')
plt.title('Document Groups by ΠΕΡΙΕΧΟΜΕΝΑ Content')
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig('plots/groups_scatter.png', dpi=300, bbox_inches='tight')
