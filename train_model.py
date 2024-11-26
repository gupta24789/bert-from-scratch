import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tqdm
import utils
from dataset import BERTDataset
from model import BERTLM, BERTModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from transformers import BertTokenizer
from torch.utils.data import DataLoader

import random
from trainer import Trainer

## Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")

## set seed
seed = 121
torch.manual_seed(seed)
if device == "cuda":
    torch.cuda.manual_seed(seed)
random.seed(seed)

## Load Config
config = utils.load_config("config.yaml")
## Load Data
pairs = utils.load_data()
## Load Tokenizer
tokenizer = BertTokenizer.from_pretrained("models")

## Train & test Split
random.shuffle(pairs)
train_size = int(len(pairs) * 0.90)
train_pairs = pairs[:train_size]
test_pairs = pairs[train_size:]
print(f"Train samples : {len(train_pairs)}")
print(f"Test pairs : {len(test_pairs)}")

## Dataset
train_ds = BERTDataset(data_pairs=train_pairs, max_len=config['max_len'], tokenizer=tokenizer)
train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
test_ds = BERTDataset(data_pairs=test_pairs, max_len=config['max_len'], tokenizer=tokenizer)
test_dl = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)

## Load Model
vocab_size = tokenizer.vocab_size
bert_model = BERTModel(config, vocab_size)
bert_lm = BERTLM(bert_model, vocab_size).to(device)

# ## Test Model
trainer = Trainer(bert_lm, tokenizer, config, device, 10, is_logging=True)
trainer.train(train_dl, test_dl)




