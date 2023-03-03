# Data Collection
This project is to collect datasets for the task of generating descriptions for charts and graphs. 
The datasets are collected from the following sources:
1. Wikipedia
2. Statista
3. Pew
4. Accessbility Journals
5. 
## Collecting the data
1. Concadia
    1. Download `wiki_split.json` and `resized.zip`. 
    2. Run `concadia.py`
    3. Filter manually via visual inspection
2. HCI
    1. Download `images.zip` and `hci.jsonl`
    2. Run `hci.py`
    3. Filter manually via visual inspection
3. Conceptual Captions
    1. Download `Train_GCC-training.tsv` file
    2. Run `conceptual_captions.py`
    3. Filter manually via visual inspection
4. Chart-to-Text - Download `chart-to-text-dataset.zip` from https://github.com/vis-nlp/Chart-to-text
    1. Statista
    2. Pew
## Formatting the Data
1. `cd models/concadia-code/code`
2 ` python3 create_input_files.py none --dataset images --root_dir ../../../datasets`