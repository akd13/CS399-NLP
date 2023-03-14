import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from collections import Counter

nltk.download('stopwords')
from nltk.corpus import stopwords

context_densenet = pd.read_csv('context_densenet.csv')
context_resnet = pd.read_csv('context_resnet.csv')
no_context_resnet = pd.read_csv('no_context_resnet.csv')
no_context_densenet = pd.read_csv('no_context_densenet.csv')

dataset = "no_context_resnet"
datasetChartTitle = "Context ResNet"
lst = no_context_resnet.label_generated.values.tolist()

# Calculate distribution of lengths

print("List ", lst)
summary_lengths = list(map(len, lst))
print("Generated lengths ", summary_lengths)

sns.histplot(summary_lengths)
plt.title('Length Distributions')
plt.xlabel('Summary lengths')
plt.ylabel('Number of summaries')
# plt.show()
plt.savefig(dataset + '_lengthdist.png')

# Plot most common words (possibly with and without stopwords - see which one is the most informative!
# Stopwords are common words that don't add information - e.g. "and", "the"

cnt = Counter()

stop_words = set(stopwords.words('english'))

for text in lst:
    for word in text.split():
        if word.lower() not in stop_words:
            cnt[word] += 1
# See most common ten words
print("Most common words ", cnt.most_common(10))
word_freq = pd.DataFrame(cnt.most_common(15),
                             columns=['words', 'count'])
word_freq.head()

fig, ax = plt.subplots(figsize=(12, 8))

# Plot horizontal bar graph
word_freq.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="green")
ax.set_title("Most Common Unigrams")
plt.savefig(dataset + '_unigrams.png')

# Visualization of top n bigrams
from collections import Counter
cnt = Counter()
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

for text in lst:
    for word in range(0, len(text.split()) - 1):
        cnt[text.split()[word] + ' ' + text.split()[word + 1]] += 1
# See most common ten bigrams
print("Most common bigrams ", cnt.most_common(10))
word_freq = pd.DataFrame(cnt.most_common(15),
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

for text in lst:
    for word in range(0, len(text.split()) - 2):
        cnt[text.split()[word] + ' ' + text.split()[word + 1] + text.split()[word + 2]] += 1
# See most common ten bigrams
print("Most common trigrams ", cnt.most_common(10))
word_freq = pd.DataFrame(cnt.most_common(15),
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