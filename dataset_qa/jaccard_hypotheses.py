import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

os.chdir('../hypotheses/')


no_context_df = pd.read_csv('no_context_statista_pew.csv', sep=',')
context_df = pd.read_csv('context_statista_pew.csv')

merged_df = pd.merge(no_context_df[['img', 'context_true','label_generated']], 
                     context_df[['img', 'label_generated']], on='img')
merged_df.columns = ['img', 'context_true','no_context_summary', 'context_summary']

no_context_similarities = []
context_similarities = []
for i, row in merged_df.iterrows():
    context = row['context_true']
    no_context_summary = row['no_context_summary']
    context_summary = row['context_summary']
    no_context_tokens = no_context_summary.split()
    context_tokens = context_summary.split()
    no_context_similarity = jaccard(context_tokens, no_context_tokens)
    context_similarity = jaccard(context_tokens, context_tokens)
    no_context_similarities.append(no_context_similarity)
    context_similarities.append(context_similarity)

plt.bar(['No Context', 'With Context'], [np.mean(no_context_similarities), np.mean(context_similarities)])
plt.xlabel('Jaccard similarity between context and summary for Pew')
plt.ylabel('Average Jaccard similarity')
plt.show()



