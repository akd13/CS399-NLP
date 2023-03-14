import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from collections import Counter
import re
from nltk.corpus import stopwords
nltk.download('stopwords')

context_densenet = pd.read_csv('context_densenet.csv')
context_resnet = pd.read_csv('context_resnet.csv')
no_context_resnet = pd.read_csv('no_context_resnet.csv')
no_context_densenet = pd.read_csv('no_context_densenet.csv')

datasets = {'context_densenet': context_densenet,
            'context_resnet': context_resnet,
            'no_context_resnet': no_context_resnet,
            'no_context_densenet': no_context_densenet}

regex_extract_word = r'\b\w+\b'
for dataset in datasets.keys():
    # dataset = "no_context_resnet"
    # datasetChartTitle = dataset.split("_")[:-1]+dataset.split("_")[-1]
    split_tokens = dataset.split("_")
    datasetChartTitle = dataset.split("_")[0] + " " + dataset.split("_")[-1] if len(split_tokens)==2 else \
        dataset.split("_")[0]+" " + dataset.split("_")[1] + " " + dataset.split("_")[-1]
    lst = datasets[dataset].label_generated.values.tolist()

    # Calculate distribution of lengths
    # print("List ", lst)
    summary_lengths = list(map(len, lst))
    # print("Generated lengths ", summary_lengths)
    sns.histplot(summary_lengths)
    plt.title('Length Distributions for ' + datasetChartTitle + ' dataset')
    plt.xlabel('Summary lengths')
    plt.ylabel('Number of summaries')
    # plt.show()
    plt.savefig(dataset + '_lengthdist.png')
    plt.close()

    # Plot most common words (possibly with and without stopwords - see which one is the most informative!
    # Stopwords are common words that don't add information - e.g. "and", "the"

    cnt = Counter()
    cnt_bigrams = Counter()
    cnt_trigrams = Counter()

    stop_words = set(stopwords.words('english'))
    for text in lst:
        words = re.findall(regex_extract_word, text)
        for i in words:
            if i.lower() not in stop_words:
                cnt[i] += 1
        for i in range(0, len(words) - 1):
            cnt_bigrams[words[i] + ' ' + words[i + 1]] += 1
        for i in range(0, len(words) - 2):
            cnt_trigrams[words[i] + ' ' + words[i + 1] + ' ' + words[i + 2]] += 1
    # See most common ten words
    print("**** DATASET {} ****".format(dataset))
    print("Most common words ", cnt.most_common(10))
    word_freq = pd.DataFrame(cnt.most_common(15), columns=['words', 'count'])
    word_freq.head()
    fig, ax = plt.subplots(figsize=(12, 8))
    # Plot horizontal bar graph
    word_freq.sort_values(by='count').plot.barh(x='words',
                          y='count',
                          ax=ax,
                          color="green")
    ax.set_title("Most Common Unigrams")
    plt.savefig(dataset + '_unigrams.png')
    plt.close()

    print("Most common bigrams ", cnt_bigrams.most_common(10))
    word_freq = pd.DataFrame(cnt_bigrams.most_common(15),
                                 columns=['words', 'count'])
    word_freq.head()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot horizontal bar graph
    word_freq.sort_values(by='count').plot.barh(x='words',
                          y='count',
                          ax=ax,
                          color="green")
    ax.set_title("Most Common Bigrams")
    plt.savefig(dataset + '_bigrams.png')
    plt.close()

    # See most common ten trigrams
    print("Most common trigrams ", cnt_trigrams.most_common(10))
    word_freq = pd.DataFrame(cnt_trigrams.most_common(15),
                                 columns=['words', 'count'])
    word_freq.head()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot horizontal bar graph
    word_freq.sort_values(by='count').plot.barh(x='words',
                          y='count',
                          ax=ax,
                          color="green")
    ax.set_title("Most Common Trigrams")
    plt.savefig(dataset + '_trigrams.png')
    plt.close()
