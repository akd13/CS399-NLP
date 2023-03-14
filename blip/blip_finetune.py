from transformers import AutoProcessor, AutoModelForCausalLM, TrainingArguments, Trainer, BlipForConditionalGeneration, BlipTextConfig, BlipVisionConfig, BlipConfig
from datasets import load_dataset, Image
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

#model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration(config)
print("Config ", model.config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

images_path = '../datasets/downsampled_images'

dataset = load_dataset("imagefolder", data_dir=images_path, split="train")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_dataset = dataset.filter(lambda example: example["split"] == "train")
v_dataset = dataset.filter(lambda example: example["split"] == "val")
test_dataset = dataset.filter(lambda example: example["split"] == "test")

print("Length of train dataset ", len(train_dataset))
print("Length of val dataset ", len(v_dataset))
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
        encoding = self.processor(images=item["image"], text=item["label"], padding="max_length", return_tensors="pt")
        # remove batch dimension


        encoding = {k:v.squeeze() for k,v in encoding.items()}

        encoded_context = tokenizer.encode_plus(
                text=item["context"],  # the sentence to be encoded
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length = 512,  # maximum length of a sentence
                pad_to_max_length=True,  # Add [PAD]s
                truncation=True,
                return_attention_mask = True,  # Generate the attention mask
                return_tensors = 'pt',  # ask the function to return PyTorch tensors
        )
        #print("Shape of encoded context ", encoded_context['input_ids'].size())
        #print("Shape of encoding input IDS ", encoding['input_ids'].size())

        encoded_context['input_ids'] = torch.squeeze(encoded_context['input_ids'])
        encoding['input_ids'] = torch.cat((encoding['input_ids'], encoded_context['input_ids']))
        attn_mask = encoded_context['attention_mask']
        return encoding

train_dataset = ContextDataset(train_dataset, feature_extractor)
valid_dataset = ContextDataset(train_dataset, feature_extractor)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1)
valid_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()

for epoch in range(1):
    print("Epoch:", epoch)
    for idx, batch in enumerate(train_dataloader):
        #print("Idx ", idx)
       # print("Batch ", batch)
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)

        #print("input ids shape ", input_ids.shape)
        #print("pixel values ", pixel_values.shape)

        if (input_ids.shape[1] > 1024): # 512 if no context dataset
            continue

        outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=input_ids)
    
        loss = outputs.loss

        print("Loss:", loss.item())

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

# prepare image for the model
# load image
example = v_dataset[10]
image = example["image"]

inputs = feature_extractor(images=image, return_tensors="pt").to(device)
pixel_values = inputs.pixel_values

generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
generated_caption = feature_extractor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Generated caption ", generated_caption)

fig = plt.figure(figsize=(18, 14))

hypotheses = []
refs = []

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

