import json
from typing import List
import numpy as np
import matplotlib.pyplot as plt

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

root = '../datasets/'


def load_data(dataset_dir: str) -> List[dict]:
    with open(f"{root}/{dataset_dir}/{dataset_dir}.json", "r") as f:
        data = json.load(f)
    return data["images"]


datasets = ["hci", "statista", "pew", "concadia"]
description_caption = []
description_context = []
description_caption_context = []
datasets_sets = {}
for dataset in datasets:
    print(f"Dataset: {dataset}")
    data = load_data(dataset)
    n = len(data)
    desc_set = set()
    cap_set = set()
    ctx_set = set()
    for i in range(n):
        desc_set.update(data[i]["description"]["tokens"])
        cap_set.update(data[i]["caption"]["tokens"])
        ctx_set.update(data[i]["context"]["tokens"])
    desc_set = list(desc_set)
    cap_set = list(cap_set)
    ctx_set = list(ctx_set)

    desc_cap_jaccard = jaccard(desc_set, cap_set)
    print(f"Jaccard similarity between description and caption: {desc_cap_jaccard:.4f}")
    description_caption.append(desc_cap_jaccard)
    desc_ctx_jaccard = jaccard(desc_set, ctx_set)
    print(f"Jaccard similarity between description and context: {desc_ctx_jaccard:.4f}")
    description_context.append(desc_ctx_jaccard)
    cap_ctx_set = set(cap_set).union(ctx_set)
    cap_ctx_list = list(cap_ctx_set)
    desc_cap_ctx_jaccard = jaccard(desc_set, cap_ctx_list)
    print(f"Jaccard similarity between description and caption+context: {desc_cap_ctx_jaccard:.4f}")
    description_caption_context.append(desc_cap_ctx_jaccard)
    datasets_sets[dataset] = {'desc': desc_set, 'cap': cap_set, 'ctx': ctx_set, 'cap_ctx': cap_ctx_set}
    print("=====")

ind = np.arange(len(datasets))
width = 0.25
fig, ax = plt.subplots()
rects1 = ax.bar(ind - width, description_caption, width, label='Description To Caption')
rects2 = ax.bar(ind, description_context, width, label='Description To Context')
rects3 = ax.bar(ind + width, description_caption_context, width, label='Description To Caption + Context')

ax.set_ylabel('Jaccard Similarity')
ax.set_title('Jaccard Similarity Comparison')
ax.set_xticks(ind)
ax.set_xticklabels(datasets)
ax.legend(loc='best', fontsize=7)

# plt.show()
plt.savefig('jaccard/jaccard_similarity_within_datasets.png')

# Compute jaccard similarity between statista and other datasets
statista = datasets_sets['statista']
dataset_jaccard_values = {}
for dataset in datasets:
    if dataset != 'statista':
        ctx_desc_jaccard = jaccard(statista['desc'], datasets_sets[dataset]['desc'])
        ctx_caption_jaccard = jaccard(statista['cap'], datasets_sets[dataset]['cap'])
        ctx_context_jaccard = jaccard(statista['ctx'], datasets_sets[dataset]['ctx'])
        caption_context_jaccard = jaccard(statista['cap_ctx'], datasets_sets[dataset]['cap_ctx'])
        print(f"Jaccard similarity between statista and {dataset} description: {ctx_desc_jaccard:.4f}")
        print(f"Jaccard similarity between statista and {dataset} caption: {ctx_caption_jaccard:.4f}")
        print(f"Jaccard similarity between statista and {dataset} context: {ctx_context_jaccard:.4f}")
        print(f"Jaccard similarity between statista and {dataset} caption+context: {caption_context_jaccard:.4f}")
        print("=====")
        dataset_jaccard_values[dataset] = [ctx_desc_jaccard, ctx_caption_jaccard, ctx_context_jaccard, caption_context_jaccard]

hci_values = dataset_jaccard_values['hci']
pew_values = dataset_jaccard_values['pew']
concadia_values = dataset_jaccard_values['concadia']

# Bar plot
x_labels = ['Description', 'Caption', 'Context', 'Caption+Context']
x = range(len(x_labels))
width = 0.2
fig, ax = plt.subplots()
ax.bar([i-width for i in x], hci_values, width, label='HCI')
ax.bar(x, pew_values, width, label='PEW')
ax.bar([i+width for i in x], concadia_values, width, label='Concadia')
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.set_ylabel('Jaccard Similarity')
ax.set_title('Jaccard Similarity between Statista and test datasets')
ax.legend()
# plt.show()
plt.savefig('jaccard/jaccard_similarity_statista_others.png')

# Compute jaccard similarity between context of statista and other datasets
statista = datasets_sets['statista']
dataset_jaccard_values = {}
for dataset in datasets:
    if dataset != 'statista':
        ctx_desc_jaccard = jaccard(statista['ctx'], datasets_sets[dataset]['desc'])
        ctx_caption_jaccard = jaccard(statista['ctx'], datasets_sets[dataset]['cap'])
        ctx_context_jaccard = jaccard(statista['ctx'], datasets_sets[dataset]['ctx'])
        print(f"Jaccard similarity between statista and {dataset} description: {ctx_desc_jaccard:.4f}")
        print(f"Jaccard similarity between statista and {dataset} caption: {ctx_caption_jaccard:.4f}")
        print(f"Jaccard similarity between statista and {dataset} context: {ctx_context_jaccard:.4f}")
        print("=====")
        dataset_jaccard_values[dataset] = [ctx_desc_jaccard, ctx_caption_jaccard, ctx_context_jaccard]

hci_values = dataset_jaccard_values['hci']
pew_values = dataset_jaccard_values['pew']
concadia_values = dataset_jaccard_values['concadia']

# Bar plot
x_labels = ['Description', 'Caption', 'Context']
x = range(len(x_labels))
width = 0.2
fig, ax = plt.subplots()
ax.bar([i-width for i in x], hci_values, width, label='HCI')
ax.bar(x, pew_values, width, label='PEW')
ax.bar([i+width for i in x], concadia_values, width, label='Concadia')
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.set_ylabel('Jaccard Similarity')
ax.set_title('Jaccard Similarity between Statista\'s Context and test datasets')
ax.legend()
# plt.show()
plt.savefig('jaccard/jaccard_similarity_statista_context_others.png')

# Compute jaccard similarity between context of pew and other datasets
pew = datasets_sets['pew']
dataset_jaccard_values = {}
for dataset in datasets:
    if dataset != 'pew':
        ctx_desc_jaccard = jaccard(pew['ctx'], datasets_sets[dataset]['desc'])
        ctx_caption_jaccard = jaccard(pew['ctx'], datasets_sets[dataset]['cap'])
        ctx_context_jaccard = jaccard(pew['ctx'], datasets_sets[dataset]['ctx'])
        print(f"Jaccard similarity between pew and {dataset} description: {ctx_desc_jaccard:.4f}")
        print(f"Jaccard similarity between pew and {dataset} caption: {ctx_caption_jaccard:.4f}")
        print(f"Jaccard similarity between pew and {dataset} context: {ctx_context_jaccard:.4f}")
        print("=====")
        dataset_jaccard_values[dataset] = [ctx_desc_jaccard, ctx_caption_jaccard, ctx_context_jaccard]

hci_values = dataset_jaccard_values['hci']
statista_values = dataset_jaccard_values['statista']
concadia_values = dataset_jaccard_values['concadia']

# Bar plot
x_labels = ['Description', 'Caption', 'Context']
x = range(len(x_labels))
width = 0.2
fig, ax = plt.subplots()
ax.bar([i-width for i in x], hci_values, width, label='HCI')
ax.bar(x, pew_values, width, label='Statista')
ax.bar([i+width for i in x], concadia_values, width, label='Concadia')
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.set_ylabel('Jaccard Similarity')
ax.set_title('Jaccard Similarity between Pew\'s Context and test datasets')
ax.legend()
# plt.show()
plt.savefig('jaccard/jaccard_similarity_pew_context_others.png')