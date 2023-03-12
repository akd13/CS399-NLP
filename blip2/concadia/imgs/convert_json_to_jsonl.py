import json

with open('concadia.json', 'r') as f:
    json_data = json.load(f)
    
print("Json data ", json_data)

with open('metadata.jsonl', 'w') as outfile:
    for entry in json_data['images']:
        print("Entry ", entry)
        entry['file_name'] = entry['filename']
        del entry['filename']

        # Maybe make the text part of it??

        entry['text'] = entry['context']['raw'] + entry['caption']['raw']
        entry['label'] = entry['description']['raw']

        # QUESTION: Do we use article ID or original filename at all?

        del entry['description']
        del entry['context']
        del entry['caption']

        json.dump(entry, outfile)
        outfile.write('\n')