import pandas as pd
from sentence_transformers import SentenceTransformer, util

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

for dataset in datasets.keys():
    datasets[dataset] = datasets[dataset].fillna('')

for dataset in ['densenet']:
    split_tokens = dataset.split("_")

    no_context_dataset = 'no_context_' + dataset 
    context_dataset = 'context_' + dataset

    hypotheses_context = datasets[context_dataset].label_generated.values.tolist()
    hypotheses_nocontext = datasets[no_context_dataset].label_generated.values.tolist()

    context = datasets[context_dataset].context_true.values.tolist()
    ground_truth = datasets[no_context_dataset].label_true.values.tolist()

    print("Generating hypotheses context embeddings")
    hypotheses_context_embeddings = model.encode(hypotheses_context, convert_to_tensor=True)

    print("Generating hypotheses no-context embeddings")
    hypotheses_no_context_embeddings = model.encode(hypotheses_nocontext, convert_to_tensor=True)

    print("Generating context embeddings")
    context_embeddings = model.encode(context, convert_to_tensor=True)

    print("Generating ground truth context embeddings")
    ground_truth_embeddings = model.encode(ground_truth, convert_to_tensor=True)

    print("Calculating cosine scores")

    context_context_cosine_scores = util.cos_sim(hypotheses_context_embeddings, context_embeddings)
    context_ground_truth_cosine_scores = util.cos_sim(hypotheses_context_embeddings, ground_truth_embeddings)

    nocontext_context_cosine_scores = util.cos_sim(hypotheses_no_context_embeddings, context_embeddings)
    nocontext_ground_truth_cosine_scores = util.cos_sim(hypotheses_no_context_embeddings, ground_truth_embeddings)

    context_cosine_scores = [float(context_cosine_scores[i][i]) for i in range(0, len(hypotheses_context))]
    ground_truth_cosine_scores = [float(ground_truth_cosine_scores[i][i]) for i in range(0, len(hypotheses_context))]

    nocontext_context_cosine_scores = [float(nocontext_context_cosine_scores[i][i]) for i in range(0, len(hypotheses_context))]
    nocontext_ground_truth_cosine_scores = [float(nocontext_ground_truth_cosine_scores[i][i]) for i in range(0, len(hypotheses_context))]

    with open('cosine_similarity_scores/context_context_cosine_scores.txt', 'w') as f:
       f.write(str(nocontext_ground_truth_cosine_scores))

    with open('cosine_similarity_scores/context_ground_truth_cosine_scores.txt', 'w') as f:
        f.write(str(nocontext_ground_truth_cosine_scores))

    with open('cosine_similarity_scores/nocontext_context_cosine_scores.txt', 'w') as f:
       f.write(str(nocontext_ground_truth_cosine_scores))

    with open('cosine_similarity_scores/nocontext_ground_truth_cosine_scores.txt', 'w') as f:
        f.write(str(nocontext_ground_truth_cosine_scores))