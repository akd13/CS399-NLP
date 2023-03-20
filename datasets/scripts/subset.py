import os
import pandas as pd

# check how many images of each type exist from file
all_images = os.listdir('../images')
datasets = ['concadia', 'hci','pew','statista']
count_map = {'concadia':0, 'hci':0,'pew':0,'statista':0}
for i in all_images:
    for d in datasets:
        if i.startswith(d):
            count_map[d]+=1
print(count_map, sum(count_map.values()), len(all_images))

# check if all entries in JSON are fine
df = pd.read_json('../images/images.json')
for i,row in df['images'].items():
    if i==0:
        print(row)
    if not os.path.isfile('images/'+row['filename']):
        print(row['filename'])
