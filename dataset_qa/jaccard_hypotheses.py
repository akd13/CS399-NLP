import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


os.chdir('../hypotheses/')

datasets = ['pew', 'hci', 'concadia']
results = []

for dataset in datasets:
    no_context_df = pd.read_csv(f'no_context_statista_{dataset}.csv', sep=',')
    context_df = pd.read_csv(f'context_statista_{dataset}.csv')

    merged_df = pd.merge(no_context_df[['img', 'context_true', 'label_generated']],
                         context_df[['img', 'label_generated']], on='img')
    merged_df.columns = ['img', 'context_true', 'no_context_summary', 'context_summary']

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

    results.append({'dataset': dataset, 'no_context_avg': np.mean(no_context_similarities),
                    'context_avg': np.mean(context_similarities)})


df = pd.DataFrame(results)
print(df.head())
df = df.rename(columns={"no_context_avg": "No Context Summary", "context_avg": "Context Summary"})
df.set_index('dataset', inplace=True)
df.plot(kind='bar')
plt.xticks(rotation=0)
plt.xlabel('Dataset')
plt.ylabel('Average Jaccard similarity')
plt.title('Jaccard Similarity of Generated Summaries, Trained on Statista')
plt.show()
plt.savefig('../dataset_qa/jaccard/jaccard_similarity_all_with_context.png')