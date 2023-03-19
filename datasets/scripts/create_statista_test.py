import json
import os
import shutil
root = '../../datasets/'
# Load the data for both datasets
with open(root+'statista/statista.json') as f:
    statista_data = json.load(f)

with open(root+'statista-large/statista-large.json') as f:
    statista_large_data = json.load(f)

# Get a list of the filenames in the original dataset
statista_filenames = [item['filename'] for item in statista_data['images']]

# Create the test dataset directory
os.makedirs(root+'statista-test/imgs', exist_ok=True)

# Copy images from statista-large to statista-test if they are not already in statista
for item in statista_large_data['images']:
    filename = item['filename']
    try:
        if filename not in statista_filenames:
            shutil.copyfile(os.path.join(root+'statista-large/imgs', filename),
                            os.path.join(root+'statista-test/imgs', filename))
    except:
        print('Error copying file: {}'.format(filename))

# Create the test dataset JSON file
test_data = {'images': []}
for item in statista_large_data['images']:
    if item['filename'] not in statista_filenames:
        test_data['images'].append(item)
with open(root+'statista-test/statista-test.json', 'w') as f:
    json.dump(test_data, f)
