import json

images_path = '../datasets/downsampled_images/'
with open(images_path + 'images.json', 'r') as f:
    json_data = json.load(f)
    
print("Json data ", json_data)

with open(images_path + 'metadata.jsonl', 'w') as outfile:
    for entry in json_data['images']:
        print("Entry ", entry)
        entry['file_name'] = entry['filename']
        del entry['filename']

        entry['text'] = entry['context']['raw'] + entry['caption']['raw']
        entry['label'] = entry['description']['raw']

        del entry['description']
        del entry['context']
        del entry['caption']

        json.dump(entry, outfile)
        outfile.write('\n')
