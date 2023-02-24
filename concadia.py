"""### View hci-charts-images and their corresponding json entry"""

import json
import os
import re
import shutil
from chart_regex import chart_regex
prefix_image = 'concadia-images/'

# Opening JSON file
f = open('concadia-images/wiki_split.json')
data = json.load(f)
f.close()
os.system('unzip resized.zip')
new_json = {'hci-charts-images': []}
saved_prefix = 'concadia-charts/'
for i, entry in enumerate(data['hci-charts-images']):
    if i % 10 == 0:
        print(i, "entries done")
    match = re.search(chart_regex, entry['description']['raw'])
    description = entry['description']['raw']
    caption = entry['caption']['raw']
    if match and description != caption:
        print(entry['orig_filename'])
        print(entry['description']['raw'])
        image_filepath = entry['filename']
        shutil.copyfile(prefix_image + image_filepath, saved_prefix + image_filepath)
        new_json['hci-charts-images'].append(entry)

with open("concadia-charts/all.json", "w") as outfile:
    json.dump(new_json, outfile)
