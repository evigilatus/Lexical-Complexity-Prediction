#!/usr/bin/env python
# coding: utf-8

# #### Imports and dataset loading


# Libraries
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import torch

# Models

import torch.nn as nn
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel

# Training

import torch.optim as optim

# Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, mean_absolute_error


train_single_tsv = '../dataset/train/lcp_single_train.tsv'
df_train_single = pd.read_csv(train_single_tsv, sep='\t', header=0, keep_default_na=False)
test_single_tsv = '../dataset/test/lcp_single_test.tsv'
df_test_single = pd.read_csv(test_single_tsv, sep='\t', header=0)


max_sent_len = 18
model_hidden_size = 768

class Roberta(nn.Module):
    def __init__(self):
        super(Roberta, self).__init__()

        self.encoder = RobertaModel.from_pretrained("roberta-base")
        self.fc1 = nn.Linear(max_sent_len * model_hidden_size//2, 200)
        self.fc2 = nn.Linear(200, 1)
        self.softmax = nn.Softmax(dim = 0) 
        

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        last_hidden_state, _ = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[:2]
        b_size = last_hidden_state.shape[0]
        seq_len = last_hidden_state.shape[1]
        model_hidden_size = last_hidden_state.shape[2]

        last_hidden_state = last_hidden_state.reshape(b_size, seq_len * model_hidden_size)

        x = self.fc1(last_hidden_state)
        x = self.fc2(x)

        return self.softmax(x)


model = Roberta()
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


#x y for word counting in sentence
list_sentences = df_train_single["sentence"].tolist()
tokens = df_train_single['token'].tolist()
# Check for max sentence length instead of hardcoded 60
# input_data = tokenizer(list_sentences, tokens, padding=True, truncation=True, max_length=max_sent_len, return_tensors='pt')
input_data = tokenizer(tokens, padding=True, truncation=True, max_length=max_sent_len, return_tensors='pt')
target_data = df_train_single['complexity']


from torch.utils.data import Dataset

class WordcountDataset(Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data

    def __len__(self):
        return len(df_train_single)

    def __getitem__(self, idx):
        input_ids = self.input_data['input_ids'][idx]
        token_type_ids = [], # self.input_data['token_type_ids'][idx]
        attention_masks = self.input_data['attention_mask'][idx]
        out = self.target_data[idx]
        
        result = {
            'input_ids': torch.from_numpy(np.array(input_ids)).long(),
            'token_ids': [], # torch.from_numpy(np.array(token_type_ids)).long(),
            'attention_mask': torch.from_numpy(np.array(attention_masks)).float(),
            'out': torch.from_numpy(np.array([out])).float()
        }
        
        return result


def train(model, input_ids, attention_mask, token_ids, y, optimizer, criterion):
    model.zero_grad()
    output = model(input_ids=input_ids, attention_mask=attention_mask) #, token_type_ids=token_ids)
    loss = criterion(output,y)
    loss.backward()
    optimizer.step()

    return loss, output


# In[ ]:


from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

criterion = nn.MSELoss()
EPOCHS = 10
BATCH_SIZE = 8
optm = Adam(model.parameters(), lr = 0.001)

dataset = WordcountDataset(input_data, target_data)
data_train = DataLoader(dataset = dataset, batch_size = BATCH_SIZE, shuffle = True)

for epoch in range(EPOCHS):
    epoch_loss = 0
    correct = 0
    
    for bidx, batch in enumerate(data_train):
        
        input_ids = batch['input_ids']
        token_ids = batch['token_ids']
        attention_mask = batch['attention_mask']
        out = batch['out']
        
        
        #start = time.time()
        loss, predictions = train(model,input_ids, attention_mask, token_ids, out, optm, criterion)
        epoch_loss+=loss
        #print("Predict time: {}".format(time.time() - start))
        
    print('Epoch {} Loss : {}'.format((epoch+1),epoch_loss))
    y_true = [test_dataset[i]['out'].item() for i in range(len(test_dataset))]
    y_pred = []

    test_loader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = True)
    for bidx, batch in enumerate(test_loader):
        #start = time.time()
        x_train = batch['inp']
        y_pred.append(net(x_train))

    y_pred = [x.item() for i in range(len(y_pred)) for x in y_pred[i] ]

    mae = mean_absolute_error(y_true, y_pred)
    print("MAE for test data: ", mae)
