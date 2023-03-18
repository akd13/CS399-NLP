import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns

context_densenet = pd.read_csv('results/context_densenet.csv')
context_resnet = pd.read_csv('results/context_resnet.csv')
no_context_resnet = pd.read_csv('results/no_context_resnet.csv')
no_context_densenet = pd.read_csv('results/no_context_densenet.csv')

datasets = {'context_densenet': context_densenet,
            'context_resnet': context_resnet,
            'no_context_resnet': no_context_resnet,
            'no_context_densenet': no_context_densenet}

for dataset in datasets.keys():
    datasets[dataset] = datasets[dataset].fillna('')

dataset = 'densenet'

context_dataset = 'context_' + dataset 
no_context_dataset = 'no_context_' + dataset 

hypotheses_context = datasets[context_dataset].label_generated.values.tolist()
hypotheses_nocontext = datasets[no_context_dataset].label_generated.values.tolist()

context = datasets[context_dataset].context_true.values.tolist()
ground_truth = datasets[no_context_dataset].label_true.values.tolist()

with open('cosine_similarity_scores/context_context_cosine_scores.txt', 'r') as f:
    data = f.readlines()[0]
    context_context_cosine_scores = data.strip('[').strip(']').split(',')

context_context_cosine_scores = [float(i) for i in context_context_cosine_scores]

with open('cosine_similarity_scores/context_ground_truth_cosine_scores.txt', 'r') as f:
    data = f.readlines()[0]
    context_ground_truth_cosine_scores = data.strip('[').strip(']').split(',')

print("Ground truth cosine scores ", context_ground_truth_cosine_scores)

context_ground_truth_cosine_scores = [float(i) for i in context_ground_truth_cosine_scores]

with open('cosine_similarity_scores/nocontext_context_cosine_scores.txt', 'r') as f:
    data = f.readlines()[0]
    nocontext_context_cosine_scores = data.strip('[').strip(']').split(',')

nocontext_context_cosine_scores = [float(i) for i in nocontext_context_cosine_scores]

with open('cosine_similarity_scores/nocontext_ground_truth_cosine_scores.txt', 'r') as f:
    data = f.readlines()[0]
    nocontext_ground_truth_cosine_scores = data.strip('[').strip(']').split(',')

nocontext_ground_truth_cosine_scores = [float(i) for i in nocontext_ground_truth_cosine_scores]

print("Sanity check: (no-context model) Plotting correlation matrix of semantic similarity to context and sem similarity to ground truth")

df = pd.DataFrame(list(zip(nocontext_context_cosine_scores, nocontext_ground_truth_cosine_scores)),
               columns =['Context', 'Ground Truth'])
    
sns.scatterplot(x='Context', y="Ground Truth", data=df)
plt.savefig(dataset + '_nocontext_cosine_similarity_correlation_plot.jpg', dpi=220, bbox_inches='tight')
plt.show()

print("Plotting correlation matrix of semantic similarity to context and sem similarity to ground truth")

df = pd.DataFrame(list(zip(context_context_cosine_scores, context_ground_truth_cosine_scores)),
               columns =['Context', 'Ground Truth'])
    
sns.scatterplot(x='Context', y="Ground Truth", data=df)
plt.savefig(dataset + '_cosine_similarity_correlation_plot.jpg', dpi=220, bbox_inches='tight')
plt.show()

print("Plotting correlation matrix of semantic similarity between context and no-context closeness to ground truth")
df = pd.DataFrame(list(zip(context_ground_truth_cosine_scores, nocontext_ground_truth_cosine_scores)),
               columns =['Context', 'Ground Truth'])
    
sns.scatterplot(x='Context', y="Ground Truth", data=df)
plt.savefig(dataset + '_cosine_similarity_correlation_plot.jpg', dpi=220, bbox_inches='tight')
plt.show()

#Find the pairs with the highest cosine similarity scores between themselves and context
print("Context model: Top three with highest similarity score to context")

pairs = []

for i in range(len(context_context_cosine_scores)):
    pairs.append({'index': i, 'score': context_context_cosine_scores[i]})

    #Sort scores in decreasing order
pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

for pair in pairs[0:3]:
    i = pair['index']
    print("Hypothesis: ", hypotheses_context[i])
    print("Context: ", context[i])
    print("Score ", context_context_cosine_scores[i])

print("Top three with lowest similarity score to context")

pairs = sorted(pairs, key=lambda x: x['score'])

for pair in pairs[0:3]:
    i = pair['index']
    print("Hypothesis: ", hypotheses_context[i])
    print("Context: ", context[i])
    print("Score ", context_context_cosine_scores[i])

print("Top three with highest similarity score to ground truth")

pairs = []
for i in range(len(context_ground_truth_cosine_scores)):
    pairs.append({'index': i, 'score': context_ground_truth_cosine_scores[i]})

    #Sort scores in decreasing order
pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

for pair in pairs[0:3]:
    i = pair['index']
    print("Hypothesis: ", hypotheses_context[i])
    print("Context: ", context[i])
    print("Score ", context_ground_truth_cosine_scores[i])

print("Top three with lowest similarity score to ground truth")

pairs = sorted(pairs, key=lambda x: x['score'])

for pair in pairs[0:3]:
    i = pair['index']
    print("Hypothesis: ", hypotheses_context[i])
    print("Context: ", context[i])
    print("Score ", context_ground_truth_cosine_scores[i])

print("No context model")

pairs = []

for i in range(len(nocontext_context_cosine_scores)):
    pairs.append({'index': i, 'score': nocontext_context_cosine_scores[i]})

    #Sort scores in decreasing order
pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

for pair in pairs[0:3]:
    i = pair['index']
    print("Hypothesis: ", hypotheses_context[i])
    print("Context: ", context[i])
    print("Score ", nocontext_context_cosine_scores[i])

print("Top three with lowest similarity score to context")

pairs = sorted(pairs, key=lambda x: x['score'])

for pair in pairs[0:3]:
    i = pair['index']
    print("Hypothesis: ", hypotheses_context[i])
    print("Context: ", context[i])
    print("Score ", nocontext_context_cosine_scores[i])

print("Top three with highest similarity score to ground truth")

pairs = []
for i in range(len(nocontext_ground_truth_cosine_scores)):
    pairs.append({'index': i, 'score': nocontext_ground_truth_cosine_scores[i]})

#Sort scores in decreasing order
pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

for pair in pairs[0:3]:
    i = pair['index']
    print("Hypothesis: ", hypotheses_context[i])
    print("Context: ", context[i])
    print("Score ", nocontext_ground_truth_cosine_scores[i])

print("Top three with lowest similarity score to ground truth")

pairs = sorted(pairs, key=lambda x: x['score'])

for pair in pairs[0:3]:
    i = pair['index']
    print("Hypothesis: ", hypotheses_context[i])
    print("Context: ", context[i])
    print("Score ", nocontext_ground_truth_cosine_scores[i])