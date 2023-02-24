import os
import json

# Take list of files sorted manually, filter it from the json, and write it out
prefix_list = ['concadia-charts', 'hci-charts']
prefix_folder = '/imgs/'
image_key_list = ['filename','local_uri']

for i,p in enumerate(prefix_list):
    prefix_dir = p + prefix_folder
    json_file = p+"/"+p + '.json'
    all_files = set(os.listdir(prefix_dir))
    filtered_json = []
    with open(json_file, 'r') as f:
        data = json.load(f)
        for d in data['images']:
            index = d[image_key_list[i]]
            if type(index) is list:
                if len(set(index)-all_files)==0:
                    filtered_json.append(d)
            elif type(index) is not list and index in all_files:
                filtered_json.append(d)
    # Writing to sample.json
    with open(p+"/"+p+"-manual-imgs.json", "w") as outfile:
        json_object = {'images':filtered_json}
        outfile.write(json.dumps(json_object))

