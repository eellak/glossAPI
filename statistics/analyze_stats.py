import pandas as pd
import matplotlib.pyplot as plt
import os

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(__file__), 'analysis_output')
os.makedirs(output_dir, exist_ok=True)

# Read the CSV file from the same directory
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'statistics.csv'))

# Calculate section statistics
total_files = len(df)
section_stats = {
    'Περίληψη': len(df[df['Περίληψη'] > 0]),
    'Βιβλιογραφία': len(df[df['Βιβλιογραφία'] > 0]),
    'ΠΕΡΙΕΧΟΜΕΝΑ': len(df[df['ΠΕΡΙΕΧΟΜΕΝΑ'] > 0]),
    'Κατάλογος Πινάκων': len(df[df['Κατάλογος Πινάκων'] > 0]),
    'ΓΛΩΣΣΑΡΙ': len(df[df['ΓΛΩΣΣΑΡΙ,  Γλωσσάρι'] > 0]),
    'EYPETHPIA': len(df[df['EYPETHPIA'] > 0])
}

# Calculate averages
averages = {
    'Avg Paragraphs': df['Total Paragraphs'].mean(),
    'Avg Lines': df['Total lines'].mean(),
    'Avg Characters': df['Total Chars'].mean()
}

# Print statistics
print(f"Total Files Analyzed: {total_files}\n")
print("Section Statistics:")
for section, count in section_stats.items():
    percentage = (count / total_files) * 100
    print(f"{section}: {count} files ({percentage:.1f}%)")

print("\nAverages:")
for metric, value in averages.items():
    print(f"{metric}: {value:.1f}")

# Create a bar plot of section statistics
plt.figure(figsize=(12, 6))
plt.bar(section_stats.keys(), section_stats.values())
plt.xticks(rotation=45, ha='right')
plt.title('Number of Files Containing Each Section')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'section_statistics.png'))

# Save detailed statistics to file
with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w') as f:
    f.write(f"Analysis Summary\n{'='*50}\n\n")
    f.write(f"Total Files Analyzed: {total_files}\n\n")
    
    f.write("Section Statistics:\n")
    f.write("-"*30 + "\n")
    for section, count in section_stats.items():
        percentage = (count / total_files) * 100
        f.write(f"{section}: {count} files ({percentage:.1f}%)\n")
    
    f.write("\nAverages:\n")
    f.write("-"*30 + "\n")
    for metric, value in averages.items():
        f.write(f"{metric}: {value:.1f}\n")
