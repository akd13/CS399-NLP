import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import base64

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print("device",device)

model_name = 'sentence-transformers/msmarco-distilbert-base-v4'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
root = '../datasets'

def compute_embeddings(text):
    with torch.no_grad():
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
        outputs = model(**inputs, output_hidden_states=True)
        embeddings = torch.mean(outputs.hidden_states[-1], dim=1).squeeze()
        return embeddings.cpu().numpy()
def cosine_similarity(e1, e2):
    dot_product = ngip.dot(e1, e2)
    norm_e1 = np.linalg.norm(e1)
    norm_e2 = np.linalg.norm(e2)
    return dot_product / (norm_e1 * norm_e2)
def compute_cosine_similarity_heatmap(data_dir):
    with open(os.path.join(root, data_dir, f"{data_dir}.json")) as f:
        data = json.load(f)

    img_embeddings = []
    text_embeddings = []
    for item in data['images']:
        img_file = os.path.join(root, data_dir, 'imgs', item['filename'])
        with open(img_file, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')
        img_embedding = compute_embeddings(img_base64)
        img_embeddings.append(img_embedding)

        text_embedding = compute_embeddings([item['description']["raw"] + item['caption']["raw"] + item['context']["raw"]])
        text_embeddings.append(text_embedding)

    img_embeddings = np.vstack(img_embeddings)
    text_embeddings = np.vstack(text_embeddings)

    cosine_similarities = []
    for i in range(len(img_embeddings)):
        cosine_similarities.append(cosine_similarity(img_embeddings[i], text_embeddings[i]))
    cosine_similarities = np.array(cosine_similarities)

    df = pd.DataFrame(cosine_similarities, columns=['cosine_similarity'])
    sns.set_theme()
    sns.displot(df, x="cosine_similarity", kind="kde", fill=True)

    plt.show()
    plt.savefig(os.path.join(root, data_dir, f"{data_dir}_heatmap.png"))


if __name__ == '__main__':
    compute_cosine_similarity_heatmap('concadia')
