import os
import csv
import ast

def parse_stats_line(line):
    # Split filename and stats
    filename, stats_str = line.split(' : ')
    
    # Convert string representation of list to actual list
    stats_list = ast.literal_eval(stats_str)
    
    # Convert list to dictionary
    stats_dict = {'filename': filename}
    for item in stats_list:
        # Clean up the key name and convert value to integer
        key = item[0].replace('##', '').strip()
        try:
            value = int(item[1])
        except (ValueError, TypeError):
            value = 0
        stats_dict[key] = value
    
    return stats_dict

def convert_stats_to_csv(input_file, output_file):
    # Read stats file and parse each line
    stats_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip empty lines and the timestamp line
            if line.strip() and not line.startswith('Processing'):
                stats_data.append(parse_stats_line(line))
    
    if not stats_data:
        print("No statistics data found!")
        return
    
    # Get all unique column names
    columns = set()
    for stats in stats_data:
        columns.update(stats.keys())
    
    # Ensure 'filename' is the first column
    columns = ['filename'] + sorted(col for col in columns if col != 'filename')
    
    # Write to CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(stats_data)

if __name__ == '__main__':
    input_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'sample_clean_output', 'stat_file.txt')
    output_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'sample_clean_output', 'statistics.csv')
    
    try:
        convert_stats_to_csv(input_file, output_file)
        print(f"Successfully converted statistics to CSV: {output_file}")
    except Exception as e:
        print(f"Error converting statistics: {e}")
