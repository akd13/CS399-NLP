import pandas as pd
from chart_regex import chart_regex
df = pd.read_csv('Train_GCC-training.tsv', delimiter='\t', header=None)
filtered_df = df[df[0].str.match(chart_regex)]
print(filtered_df.iloc[2][0],filtered_df.iloc[2][1])
filtered_df.to_csv('conceptual_captions.csv',delimiter='\t')