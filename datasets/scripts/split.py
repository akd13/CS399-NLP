import json
import random

# Load the JSON data
with open('../downsampled_images/images.json', 'r') as f:
    data = json.load(f)

# Shuffle the data
data=data['images']
random.shuffle(data)

# Split the data
num_entries = len(data)
train_split = int(num_entries * 0.8)
val_split = int(num_entries * 0.9)
train_data = data[:train_split]
val_data = data[train_split:val_split]
test_data = data[val_split:]

# Add a "split" key to each entry
for entry in train_data:
    entry['split'] = 'train'
    entry['caption']['raw'].replace('                                                 ','')
for entry in val_data:
    entry['split'] = 'val'
    entry['caption']['raw'].replace('                                                ','')
for entry in test_data:
    entry['split'] = 'test'
    entry['caption']['raw'].replace('                                                ','')

# Write the split data to separate JSON files
with open('../downsampled_images/split_images.json', 'w') as f:
    json.dump({'images':train_data+test_data+val_data}, f)
