import json

datasets = ['pew', 'hci', 'concadia', 'statista-large','statista']
for dataset in datasets:
    with open("../" + dataset + "/" + dataset + '.json', 'r') as f:
        json_data = json.load(f)

    with open("../" + dataset + '/imgs/metadata.jsonl', 'w') as outfile:
        for entry in json_data['images']:
            entry['file_name'] = entry['filename']
            del entry['filename']
            entry['context'] = entry['context']['raw'] + entry['caption']['raw']
            entry['label'] = entry['description']['raw']
            del entry['description']
            del entry['caption']
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')
