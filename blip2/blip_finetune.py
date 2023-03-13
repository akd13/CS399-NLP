from transformers import AutoProcessor, AutoModelForCausalLM, TrainingArguments, Trainer, BlipForConditionalGeneration
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

feature_extractor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Get the data and make it into a dataset object
images_path = '../datasets/downsampled_images'

dataset = load_dataset("imagefolder", data_dir=images_path, split="train")

train_dataset = dataset.filter(lambda example: example["split"] == "train")
v_dataset = dataset.filter(lambda example: example["split"] == "val")
test_dataset = dataset.filter(lambda example: example["split"] == "test")

print("Length of train dataset ", len(train_dataset))
print("Length of val dataset ", len(v_dataset))
print("Length of test dataset ", len(test_dataset))

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], text=item["text"], padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding
    
train_dataset = ImageCaptioningDataset(train_dataset, feature_extractor)
valid_dataset = ImageCaptioningDataset(train_dataset, feature_extractor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1)
valid_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()

for epoch in range(10):
  print("Epoch:", epoch)
  for idx, batch in enumerate(train_dataloader):
    input_ids = batch.pop("input_ids").to(device)
    pixel_values = batch.pop("pixel_values").to(device)

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
''' example = v_dataset[10]
image = example["image"]

inputs = feature_extractor(images=image, return_tensors="pt").to(device)
pixel_values = inputs.pixel_values

generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
generated_caption = feature_extractor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Generated caption ", generated_caption)

fig = plt.figure(figsize=(18, 14)) '''
generation_blip_finetuned = {}

for i, example in enumerate(test_dataset):
  image = example["image"]
  reference = example["text"] # NOTE: This is the example of generated caption!
  inputs = feature_extractor(images=image, return_tensors="pt").to(device)
  pixel_values = inputs.pixel_values

  generated_ids = model.generate(pixel_values=pixel_values, max_length=200)
  generated_caption = feature_extractor.batch_decode(generated_ids, skip_special_tokens=True)[0]
  generation_blip_finetuned[reference] = generated_caption

  fig.add_subplot(1, 1, i+1)
  plt.imshow(image)
  plt.axis("off")
  plt.title(f"Generated caption: {generated_caption}")

json_object = json.dumps(generation_blip_finetuned)

with open("blip_finetune_captions_test.json", "w") as outfile:
    outfile.write(json_object)
