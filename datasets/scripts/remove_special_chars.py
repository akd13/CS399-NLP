import re
import json
from unicode_to_ascii import unicode_to_ascii
from nltk.tokenize import word_tokenize

root = 'datasets/'
filenames = ['hci/hci.json','concadia/concadia.json',
             'pew/pew.json','statista/statista.json']
unicode_regex = re.compile(r'\\u[0-9a-fA-F]{4}')

def replace_unicode(match):
    unicode_char = match.group()
    ascii_char = unicode_to_ascii.get(unicode_char)
    if ascii_char:
        return ascii_char
    else:
        return unicode_char

for f in filenames:
    file = root + f
    with open(file, 'r', encoding='utf-8') as input_file:
        input_text = json.load(input_file)
        json_str = unicode_regex.sub(replace_unicode, json.dumps(input_text, ensure_ascii=False))
        json_str = json_str.replace('­ ','') #extra word to replace for soft hyphen
        input_text = json.loads(json_str)
    for image in input_text['images']:
        image['caption']['tokens'] = word_tokenize(image['caption']['raw'])
        image['description']['tokens'] = word_tokenize(image['description']['raw'])
        image['context']['tokens'] = word_tokenize(image['context']['raw'])
    with open(file.split('.')[0]+'_remove_unicode.json', 'w', encoding='utf-8') as f_new:
        json.dump(input_text, f_new, ensure_ascii=False)
