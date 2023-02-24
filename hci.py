import json
import re
import shutil
from chart_regex import chart_regex

prefix_image = 'hci-images/'

# Opening JSON file
with open('hci-images/hci.jsonl') as f:
    data = [json.loads(line) for line in f]

print(data[0])
new_json = {'hci-charts-images':[]}
saved_prefix = 'hci-charts/'
for i,entry in enumerate(data):
  if i%10==0:
    print(i, "entries done")
  match = re.search(chart_regex, entry['alt_text'])
  if match:
    print(entry['alt_text'])
    images = entry['local_uri']
    for image in images:
        shutil.copyfile(prefix_image+image, saved_prefix+image)
    new_json['hci-charts-images'].append(entry)

with open("hci-charts/hci-charts.json", "w") as outfile:
  json.dump(new_json, outfile)
