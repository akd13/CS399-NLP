import os
import json
import pandas as pd

downsampled_images = set(os.listdir('../downsampled_images'))

df = pd.read_json('../images/images.json')

filtered_rows = []
for i, row in df['images'].items():
    if row['filename'] in downsampled_images:
        filtered_rows.append(row)
        row['caption']['raw'].replace('                                                ', '')

print(len(filtered_rows))

with open('../downsampled_images/images.json', 'w') as f:
    json.dump({'images': filtered_rows}, f)