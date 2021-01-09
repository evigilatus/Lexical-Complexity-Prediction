#!/usr/bin/env python
# coding: utf-8

# Libraries

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

# Models

import torch.nn as nn
from transformers import BertTokenizer, BertModel

# Training

import torch.optim as optim

# Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[9]:


train_single_tsv = 'dataset/train/lcp_single_train.tsv'
df_train_single = pd.read_csv(train_single_tsv, sep='\t', header=0)


# In[11]:


print("Data columns: \n")
print(df_train_single.columns)
print("Total corpus len: {}".format(len(df_train_single)))
print("Subcorpus len:\n")
print(df_train_single['corpus'].value_counts())


# In[13]:


class Bert(nn.Module):

    def __init__(self):
        super(Bert, self).__init__()
        
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.fc1 = nn.Linear(9216, 1)
        self.softmax = nn.Softmax(dim = 0) 
        

    def forward(self, input_ids, attention_mask, token_type_ids):
        last_hidden_state, _ = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[:2]
        # TODO Add fc1
        flatten_state = torch.flatten(last_hidden_state)
        return self.softmax(flatten_state.float())



bert = Bert()
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# tokenizer('Test me you mf bitch!', return_tensors='pt')

# encoder = BertModel.from_pretrained("bert-base-uncased")
# r = tokenizer('Test me you mf bitch!', return_tensors='pt')
# print(r['input_ids'][0].shape)
# print(r['attention_mask'][0].shape)
# print(r['token_type_ids'][0].shape)
# encoder(r['input_ids'][0], r['attention_mask'][0], r['token_type_ids'][0])

# first, second = encoder(**tokenizer('Test me you mf bitch!', return_tensors='pt'), return_dict=True)[:2]
# print("first", first.shape, sep="\n")

# In[17]:


torch_t = torch.tensor([[1,2,5,3], [1,2,2, 5]])
flatten_t = torch.flatten(torch_t).float()
sm = nn.Softmax(dim=0)

print(torch_t)
print(flatten_t)
print(torch.randn(2, 3))
sm(flatten_t)


# In[8]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokens = tokenizer(["This is a sentence. token"])

input_ids = tokens["input_ids"]
tokenizer.convert_ids_to_tokens(input_ids[0])

print("Tokens: ", tokens)


# In[18]:


print(len(df_train_single["token"].tolist()) == len(df_train_single["sentence"].tolist()))

tokens = df_train_single["token"].tolist()

print([i for i in range(len(tokens)) if type(tokens[i]) != str])
print(tokens[3726])

#tokenizer(df_train_single["sentence"].tolist(), df_train_single["token"].tolist())
#decoded = tokenizer.decode(encoded_dict["input_ids"])


# In[21]:


#x y for word counting in sentence
list_sentences = df_train_single["sentence"].tolist()
max_sent_len = 60
x = tokenizer(list_sentences, padding=True, truncation=True, max_length=max_sent_len, return_tensors='pt')
y = [min(len(i.split()), max_sent_len) for i in list_sentences]


# In[126]:


print(x.keys())
print(list_sentences[1])
print(x['input_ids'][1])
print(len(x['input_ids'][1]))
print(y[1])

# len(bert.forward(x['input_ids'][0], x['attention_mask'][0], x['token_type_ids'][0]))



print(x['input_ids'][0].shape)
print(x['attention_mask'][0].shape)
print(x['token_type_ids'][0].shape)


# In[7]:


from torch.utils.data import Dataset

class WordcountDataset(Dataset):
    def __init__(self):
        pass
    
    def __len__(self):
        return len(df_train_single)
    
    def __getitem__(self, idx):
        input_ids = x['input_ids'][idx]
        token_type_ids = x['token_type_ids'][idx]
        attention_masks = x['attention_mask'][idx]
        out = y[idx]
        
        
        result = {
            'input_ids': torch.from_numpy(np.array(input_ids)).long(),
            'token_ids': torch.from_numpy(np.array(token_type_ids)).long(),
            'attention_mask': torch.from_numpy(np.array(attention_masks)).float(),
            'out': torch.from_numpy(np.array([out])).float()
        }
        
        #print("Idx {} fetch time: {}".format(idx, time.time() - start))
        return result

dataset = WordcountDataset()
print(dataset[500])


# In[214]:


def train(model, input_ids, attention_mask, token_ids, y, optimizer, criterion):
    model.zero_grad()
    output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_ids)
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
EPOCHS = 30
BATCH_SIZE = 64
optm = Adam(bert.parameters(), lr = 0.001)

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
        loss, predictions = train(bert, input_ids, attention_mask, token_ids, out, optm, criterion)
        epoch_loss+=loss
        #print("Predict time: {}".format(time.time() - start))
        
    print('Epoch {} Loss : {}'.format((epoch+1),epoch_loss))

