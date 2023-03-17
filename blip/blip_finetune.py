from transformers import AutoProcessor, AutoModelForCausalLM, TrainingArguments, Trainer, BlipForConditionalGeneration, BlipTextConfig, BlipVisionConfig, BlipConfig
from datasets import load_dataset, concatenate_datasets, Image
import requests
from PIL import Image
import torch
import numpy as np
import pandas as pd
from torchmetrics import BLEUScore
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
import itertools
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import json
from transformers import BertTokenizer, BertModel
from nlgeval import compute_metrics
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

feature_extractor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

config_text = BlipTextConfig(max_position_embeddings=1024)
config_vision = BlipVisionConfig()
config = BlipConfig.from_text_vision_configs(config_text, config_vision)

import torch
torch.cuda.empty_cache()

#model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration(config)
#print("Config ", model.config)

def transforms(examples):
    examples["image"] = [image.convert("RGB").resize((100,100)) for image in examples["image"]]
    return examples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

statista_images_path = '../datasets/statista/imgs'
pew_images_path = '../datasets/pew/imgs'
hci_images_path = '../datasets/hci/imgs'
concadia_images_path = '../datasets/concadia/imgs'

statista_dataset = load_dataset("imagefolder", data_dir=statista_images_path, split="train")
pew_dataset = load_dataset("imagefolder", data_dir=pew_images_path, split="train")
hci_dataset = load_dataset("imagefolder", data_dir=hci_images_path, split="train")
concadia_dataset = load_dataset("imagefolder", data_dir=concadia_images_path, split="train")

test_dataset = concatenate_datasets([pew_dataset, hci_dataset, concadia_dataset])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
statista_dataset = statista_dataset.map(transforms, batched=True)
test_dataset = test_dataset.map(transforms, batched=True)

train_dataset = statista_dataset.filter(lambda example: example["split"] == "train")
valid_dataset = statista_dataset.filter(lambda example: example["split"] == "val")
#test_dataset = dataset.filter(lambda example: example["split"] == "test")

print("Length of train dataset ", len(train_dataset))
print("Length of val dataset ", len(valid_dataset))
print("Length of test dataset ", len(test_dataset))

class NoContextDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
        self.hasContext = False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], text=item["label"], padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding
    
class ContextDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
        self.hasContext = True

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        tokenized_context = item["context"]
        image = item["image"]
        #print("Image size ", image.size)

        encoding = self.processor(images=item["image"], text=item["label"], padding="max_length", return_tensors="pt")
        # remove batch dimension

        encoding = {k:v.squeeze() for k,v in encoding.items()}
    #    print("Shape of input IDs ", encoding['input_ids'].shape)

        encoded_context = tokenizer.encode_plus(
                text=item["context"],  # the sentence to be encoded
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length = 512,  # maximum length of a sentence
                pad_to_max_length=True,  # Add [PAD]s
                truncation=True,
                return_attention_mask = True,  # Generate the attention mask
                return_tensors = 'pt',  # ask the function to return PyTorch tensors
        )
        encoded_context['input_ids'] = torch.squeeze(encoded_context['input_ids'])
        encoding['input_ids'] = torch.cat((encoding['input_ids'], encoded_context['input_ids']))
        attn_mask = encoded_context['attention_mask']
        return encoding

train_dataset = ContextDataset(train_dataset, feature_extractor)
valid_dataset = ContextDataset(valid_dataset, feature_extractor)

train_dataset = [image for image in train_dataset if image['input_ids'].shape[0] == 1024]
valid_dataset = [image for image in valid_dataset if image['input_ids'].shape[0] == 1024]
#test_dataset = [image for image in test_dataset if image['input_ids'].shape[0] == 1024]
#train_dataset = train_dataset.filter(lambda example: example["input_ids"].shape[0] > 1024)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4)
valid_dataloader = DataLoader(valid_dataset, shuffle=True, batch_size=4)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()

'''for idx, batch in enumerate(train_dataloader):
        #print("Idx ", idx)
       # print("Batch ", batch)
    input_ids = batch.pop("input_ids").to(device)
    pixel_values = batch.pop("pixel_values").to(device)
    if (input_ids.shape[1] > 1024):
        print("Input IDS shape ", input_ids.shape[1])
   ''' 

for epoch in range(1):
    print("Epoch:", epoch)
    for idx, batch in enumerate(train_dataloader):
        if (idx % 100 == 0):
            print("On " + str(epoch) + " epoch and " + str(idx) + " iteration")

        #print("Idx ", idx)
       # print("Batch ", batch)
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)

        #print("Input IDS ", input_ids)
        #print("Input IDS shape ", input_ids.shape)

        #print("input ids shape ", input_ids.shape)
        #print("pixel values ", pixel_values.shape)

        if (input_ids.shape[1] > 1024): # 512 if no context dataset
            continue

        outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=input_ids)
    
        loss = outputs.loss

#        print("Loss:", loss.item())

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

# prepare image for the model
# load image
#example = valid_dataset[10]
#image = example["image"]

#inputs = feature_extractor(images=image, return_tensors="pt").to(device)
#pixel_values = inputs.pixel_values

#generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
#generated_caption = feature_extractor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#print("Generated caption ", generated_caption)

#fig = plt.figure(figsize=(18, 14))

hypotheses = []
refs = []

#test_dataset = [image for image in test_dataset if image['input_ids'].shape[0] == 1024]

for i, example in enumerate(test_dataset):
  image = example["image"]
  label = example["label"]
  refs.append(label)

  inputs = feature_extractor(images=image, return_tensors="pt").to(device)
  pixel_values = inputs.pixel_values
 
  generated_ids = model.generate(pixel_values=pixel_values, max_length=200)
  generated_caption = feature_extractor.batch_decode(generated_ids, skip_special_tokens=True)[0]
  hypotheses.append(generated_caption)
#  fig.add_subplot(1, 1, i+1)
#  plt.imshow(image)
#  plt.axis("off")
#  plt.title(f"Generated caption: {generated_caption}")

print("Hypotheses ", hypotheses)
print("Refs ", refs)

with open("hypothesis.txt", "w") as outfile:
    outfile.write(str(hypotheses))

with open("refs.txt", "w") as outfile:
    outfile.write(str(refs))

metrics_dict = compute_metrics(hypothesis='hypothesis.txt',
                               references=['refs.txt'])

print(metrics_dict)

