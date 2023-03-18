from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import numpy as np

# plot PCA of the hypotheses context semantic embeddings with the context model
# plot PCA of the hypotheses-no context semantic embeddings with the no-context model

from sentence_transformers import SentenceTransformer

context_densenet = pd.read_csv('results/context_densenet.csv')
context_resnet = pd.read_csv('results/context_resnet.csv')
no_context_resnet = pd.read_csv('results/no_context_resnet.csv')
no_context_densenet = pd.read_csv('results/no_context_densenet.csv')

datasets = {'context_densenet': context_densenet,
            'context_resnet': context_resnet,
            'no_context_resnet': no_context_resnet,
            'no_context_densenet': no_context_densenet}

regex_extract_word = r'\b\w+\b'

model = SentenceTransformer('all-MiniLM-L6-v2')

hypotheses_context = context_densenet.label_generated.values.tolist()
hypotheses_nocontext = no_context_densenet.label_generated.values.tolist()

print("Generating hypotheses context embeddings")
hypotheses_context_embeddings = model.encode(hypotheses_context, convert_to_tensor=True)

print("Generating hypotheses no-context embeddings")
hypotheses_no_context_embeddings = model.encode(hypotheses_nocontext, convert_to_tensor=True)

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

print("Plotting correlation matrix of semantic similarity to context and sem similarity to ground truth")

sc = StandardScaler()
sc.fit(hypotheses_context_embeddings)

X_std = sc.transform(hypotheses_context_embeddings)

pca = PCA(n_components=5)
pca.fit(X_std)

print("Explained variance ratio: ", pca.explained_variance_ratio_)

plt.title("PCA for Hypotheses Embeddings (context model)")

plt.bar(range(0, 5), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(0, 5), np.cumsum(pca.explained_variance_ratio_), where='mid', label='Cumulative explained variance')

plt.legend(loc='best')

plt.xlabel("Component")
plt.ylabel("Explained Variance Ratio")

plt.savefig("hypotheses_context_pca" + ".png")

plt.close()

sc = StandardScaler()
sc.fit(hypotheses_no_context_embeddings)

X_std = sc.transform(hypotheses_context_embeddings)

pca = PCA(n_components=5)
pca.fit(X_std)

print("Explained variance ratio: ", pca.explained_variance_ratio_)

plt.title("PCA for Hypotheses Embeddings (no context model)")

plt.bar(range(0, 5), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(0, 5), np.cumsum(pca.explained_variance_ratio_), where='mid', label='Cumulative explained variance')

plt.legend(loc='best')

plt.xlabel("Component")
plt.ylabel("Explained Variance Ratio")

plt.savefig("hypotheses_no_context_pca" + ".png")

plt.close()