import json

images_path = '../datasets/hci/'
with open(images_path + 'hci.json', 'r') as f:
    json_data = json.load(f)
    
with open(images_path + 'imgs/metadata.jsonl', 'w') as outfile:
    for entry in json_data['images']:
        print("Entry ", entry)
        entry['file_name'] = entry['filename']
        del entry['filename']

        entry['context'] = entry['context']['raw'] + entry['caption']['raw']
        entry['label'] = entry['description']['raw']

        del entry['description']
        del entry['caption']

        json.dump(entry, outfile)
        outfile.write('\n')
