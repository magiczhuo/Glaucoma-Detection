import os

def get_folder_stats(root_path):
    stats = {}
    if not os.path.exists(root_path):
        print(f'Path {root_path} does not exist.')
        return stats
    
    # Normalize root_path to avoid issues with trailing slashes
    root_path = os.path.normpath(root_path)
        
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Get relative path from the root
        rel_path = os.path.relpath(dirpath, root_path)
        if rel_path == '.':
            continue
            
        # Count only files, ignore hidden files if necessary (optional)
        file_count = len([f for f in filenames if not f.startswith('.')])
        stats[rel_path] = file_count
    return stats

path1 = 'SMDG_test/test'
path2 = 'retfound_features_smdg/test'

print(f'Analyzing {path1}...')
stats1 = get_folder_stats(path1)
print(f'Analyzing {path2}...')
stats2 = get_folder_stats(path2)

all_keys = set(stats1.keys()) | set(stats2.keys())
sorted_keys = sorted(list(all_keys))

print('\nComparison Results:')
print(f'{'Subfolder':<40} | {path1:<20} | {path2:<20} | {'Match':<10}')
print('-' * 100)

match_count = 0
mismatch_count = 0

for key in sorted_keys:
    count1 = stats1.get(key, 'N/A')
    count2 = stats2.get(key, 'N/A')
    
    match = 'Yes' if count1 == count2 else 'No'
    if match == 'Yes':
        match_count += 1
    else:
        mismatch_count += 1
        
    print(f'{key:<40} | {str(count1):<20} | {str(count2):<20} | {match:<10}')

print('-' * 100)
print(f'Total folders checked: {len(sorted_keys)}')
print(f'Matching folders: {match_count}')
print(f'Mismatched folders: {mismatch_count}')