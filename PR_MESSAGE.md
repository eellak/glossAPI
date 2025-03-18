# Fix annotation process for small datasets without boundary markers

## Problem
When using `fully_annotate=True` in the `corpus.annotate()` method with small datasets, an error occurs: "All objects passed were None". This happens during the full annotation process after the classification step.

The issue occurs because:
1. The `fully_annotate_text_group` method requires both 'π' and 'β' boundary markers to be present in the dataset
2. If these markers are missing, it returns `None` instead of a processed DataFrame
3. With small test datasets, all document groups might return `None`
4. When trying to concatenate an empty list of DataFrames, the error "All objects passed were None" is raised

## Solution
This PR addresses the issue in two ways:

1. Modified the `fully_annotate_text_group` method to handle cases where boundary markers are missing:
   - Instead of returning `None`, it now applies a default annotation (marking everything as 'κ')
   - Added warning logs to indicate missing boundary markers
   - Returns the group with default annotations

2. Updated the `fully_annotate` method to handle the case when no groups are returned:
   - Added explicit check for empty `updated_groups`
   - Creates an empty DataFrame with the same columns as the input when no groups are processed
   - Added warning logs to indicate that no document groups were successfully annotated

## Benefits
- Improves robustness of the annotation process for small datasets
- Provides more informative error messages
- Maintains backward compatibility with existing code
- Makes the library more user-friendly for testing and development

## Testing
Tested with small datasets that don't contain boundary markers, confirming that the annotation process now completes without errors.
