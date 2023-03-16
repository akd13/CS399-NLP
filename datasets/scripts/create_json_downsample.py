import os
import json
import pandas as pd

downsampled_images = set(os.listdir('statista-large/imgs/'))
print(len(downsampled_images))
df = pd.read_json('statista-large/statista-large.json')

filtered_rows = []
for i, row in df['images'].items():
    if row['filename'] in downsampled_images:
        filtered_rows.append(row)
        row['caption']['raw'].replace('                                                ', '')

print(len(filtered_rows))

with open('statista/statista.json', 'w') as f:
    json.dump({'images': filtered_rows}, f, ensure_ascii=False)