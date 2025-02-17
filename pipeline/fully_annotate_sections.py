import pandas as pd
import numpy as np
import os

# ----------------------------
# Read the annotated Parquet file
# ----------------------------
input_file = "/mnt/data/sections_for_annotation_predictions.parquet"
# Read the Parquet file, using predicted_section instead of label
df = pd.read_parquet(input_file, columns=['id', 'row_id', 'filename', 'has_table', 'has_list', 'header', 'place', 'section', 'predicted_section'])

# ----------------------------
# Counters and container for results
# ----------------------------
files_missing_boundaries = 0  # Count files without both a π and a β boundary
updated_groups = []  # To collect processed groups

# ----------------------------
# Process each file group by filename
# ----------------------------
for filename, group in df.groupby('filename'):
    # Sort sections by id (which reflects absolute order)
    group = group.sort_values('id').copy()

    # Check if both boundary labels are present
    has_pi = (group['predicted_section'] == 'π').any()
    has_beta = (group['predicted_section'] == 'β').any()
    
    if has_pi and has_beta:
        # The first occurrence of 'π' defines the upper boundary
        first_pi_id = group.loc[group['predicted_section'] == 'π', 'id'].iloc[0]
        # The last occurrence of 'β' defines the lower boundary
        last_beta_id = group.loc[group['predicted_section'] == 'β', 'id'].iloc[-1]
        
        # New condition: Check if markers are in the correct order
        if first_pi_id > last_beta_id:
            files_missing_boundaries += 1
            continue  # Skip this file's annotation
        
        # Create a boolean mask for rows with label "άλλο" (i.e. not yet fully annotated)
        mask = group['predicted_section'] == 'άλλο'
        
        # Define conditions (vectorized) on the 'id' of each row
        cond_intro = group['id'] < first_pi_id    # Before the first 'π'
        cond_appendix = group['id'] > last_beta_id  # After the last 'β'
        
        # For rows with label "άλλο", assign:
        #   'ε.σ.' if the row is before the first π,
        #   'α' if the row is after the last β,
        #   and 'κ' otherwise.
        new_labels = np.select(
            [cond_intro, cond_appendix],
            ['ε.σ.', 'α'],
            default='κ'
        )
        # Update only the rows with label "άλλο"
        group.loc[mask, 'predicted_section'] = new_labels[mask]
    else:
        files_missing_boundaries += 1  # Count files missing one or both boundaries

    updated_groups.append(group)

# ----------------------------
# Concatenate updated groups back into a single DataFrame
# ----------------------------
df_updated = pd.concat(updated_groups)

# ----------------------------
# Save to a new Parquet file:
# Replace "_annotated" in the filename with "_fully_annotated"
# ----------------------------
output_file = input_file.replace("predictions", "fully_annotated")
df_updated.to_parquet(output_file, index=False)

# ----------------------------
# Print summary statistics
# ----------------------------
print("Processing complete.")
print(f"Files missing one or both boundaries (π and β): {files_missing_boundaries}")
