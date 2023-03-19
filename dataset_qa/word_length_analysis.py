import json
import os
from collections import Counter

from scipy.spatial.distance import jaccard


# Function to calculate jaccard similarity
def jaccard_similarity(set1, set2):
    return 1 - jaccard(set1, set2)

root = '../datasets/'
directories = ['hci', 'statista', 'pew', 'concadia']

for directory in directories:
    json_path = os.path.join(root,directory, directory+'.json')
    img_path = os.path.join(root, directory, 'imgs')
    with open(json_path, 'r') as f:
        data = json.load(f)

    num_images = len(data['images'])

    vocab_counter = Counter()

    desc_len = 0
    caption_len = 0
    context_len = 0

    desc_word_len = 0
    caption_word_len = 0
    context_word_len = 0

    for img in data['images']:
        # Get description, caption, and context
        desc = img['description']['raw']
        cap = img['caption']['raw']
        cont = img['context']['raw']

        # Tokens
        desc_tokens = img['description']['tokens']
        cap_tokens = img['caption']['tokens']
        cont_tokens = img['context']['tokens']

        # Sum num tokens of each type
        desc_len += len(desc_tokens)
        caption_len += len(cap_tokens)
        context_len += len(cont_tokens)

        # Sum length of each word
        desc_word_len += sum([len(word) for word in desc_tokens])
        caption_word_len += sum([len(word) for word in cap_tokens])
        context_word_len += sum([len(word) for word in cont_tokens])

        # Update vocab counters
        vocab_counter.update(set(desc_tokens + cap_tokens + cont_tokens))

    # Get avg number of tokens per type
    num_desc_words = desc_len / num_images
    num_cap_words = caption_len / num_images
    num_cont_words = context_len / num_images

    # Get avg length of words
    avg_desc_word_len = desc_word_len / desc_len
    avg_cap_word_len = caption_word_len / caption_len
    avg_cont_word_len = context_word_len / context_len

    # Get vocab size
    vocab_size = len(vocab_counter)

    # Print results
    print(directory)
    print("Number of images: ", num_images)
    print("Number of description words: ", num_desc_words)
    print("Number of caption words: ", num_cap_words)
    print("Number of context words: ", num_cont_words)
    print("Average description word length: ", avg_desc_word_len)
    print("Average caption word length: ", avg_cap_word_len)
    print("Average context word length: ", avg_cont_word_len)
    print("Vocab size: ", vocab_size)

    # for other_dir in directories:
    #     if directory != other_dir:
    #         other_json_path = os.path.join(root, other_dir, other_dir+'.json')
    #         with open(other_json_path, 'r') as f:
    #             other_data = json.load(f)
    #
    #         other_desc_counter = Counter()
    #         other_cap_counter = Counter()
    #         other_cont_counter = Counter()
    #         other_vocab_counter = Counter()
    #
    #         for img in other_data['images']:
    #             other_desc = img['description']['raw']
    #             other_cap = img['caption']['raw']
    #             other_cont = img['context']['raw']
    #
    #             other_desc_tokens = word_tokenize(other_desc.lower())
    #             other_cap_tokens = word_tokenize(other_cap.lower())
    #             other_cont_tokens = word_tokenize(other_cont.lower())

