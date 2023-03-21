# Set up env
1. Navigate to the root directory of the project - `cd CS399-NLP/`
1. Create a virtual environment
    2. `pip install virtualenv`
    2. `virtualenv venv` 
2. Activate the virtual environment - `source venv/bin/activate`
3. Install the requirements - `pip install -r requirements.txt`

# Data Collection
This project is to collect datasets for the task of generating descriptions for charts and graphs. 
The datasets are collected from the following sources:
1. Wikipedia (Concadia)
2. Statista
3. Pew
4. Accessbility Journals (HCI)
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
1. Navigate to inside the project, and `cd models/concadia-code/code`
2. To create embeddings, ensure that your dataset has all images inside directory `images` and has the metadata in 
`images.json`, then do the following - 
    3. No Context` python3 create_input_files.py none --dataset <dataset> --root_dir ../../../datasets`
    4. Context `python3 create_input_files.py context --dataset <dataset> --root_dir ../../../datasets` 
5. To train the model, do the following - 
    6. No context `python3 train.py --image_encoder_type densenet --context_encoder_type roberta description context none --epochs 10 --blank_context`
