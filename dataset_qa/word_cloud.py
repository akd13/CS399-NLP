import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
root = '../datasets/'
directories = ['hci','pew','statista','concadia']
for directory in directories:
    data_path = f'{directory}.json'
    with open(os.path.join(root, directory,data_path), 'r') as f:
        data = json.load(f)

    text = ""
    for entry in data['images']:
        text += entry['description']['raw'] + " "
        text += entry['caption']['raw'] + " "
        text += entry['context']['raw'] + " "

    wordcloud = WordCloud(width=800, height=800, background_color='white', stopwords=stopwords.words('english'),
                          min_font_size=10).generate(text)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    plt.savefig(os.path.join('wordclouds',f'word_cloud_{directory}.png'))
