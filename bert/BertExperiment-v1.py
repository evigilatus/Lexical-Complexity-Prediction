#!/usr/bin/env python
# coding: utf-8

# #### Imports and dataset loading

# In[1]:


# Libraries
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


# In[74]:


train_single_tsv = '../dataset/train/lcp_single_train.tsv'
df_train_single = pd.read_csv(train_single_tsv, sep='\t', header=0)

# Filtering null values
filtered_df = df_train_single[df_train_single['token'].notnull()]
print('Initial dataframe length: ', len(df_train_single))
print('Removing rows: \n', df_train_single.merge(filtered_df, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only'])
df_train_single = filtered_df
print('\nDataframe length after filtering null values: ', len(df_train_single))


max_sent_len = 60
bert_hidden_size = 768

class Bert(nn.Module):

    def __init__(self):
        super(Bert, self).__init__()



        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.fc1 = nn.Linear(max_sent_len * bert_hidden_size, 200)
        self.fc2 = nn.Linear(200, 1)
        self.softmax = nn.Softmax(dim = 0) 
        

    def forward(self, input_ids, attention_mask, token_type_ids):
        last_hidden_state, _ = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[:2]
        b_size = last_hidden_state.shape[0]
        seq_len = last_hidden_state.shape[1]
        bert_hidden_size = last_hidden_state.shape[2]

        last_hidden_state = last_hidden_state.reshape(b_size, seq_len * bert_hidden_size)

        x = self.fc1(last_hidden_state)
        x = self.fc2(x)

        return self.softmax(x)

    

bert = Bert()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer('Test me you mf bitch!', return_tensors='pt')


# In[77]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


#x y for word counting in sentence
list_sentences = df_train_single["sentence"].tolist()
tokens = df_train_single['token'].tolist()
# Check for max sentence length instead of hardcoded 60
input_data = tokenizer(list_sentences, tokens, padding=True, truncation=True, max_length=max_sent_len, return_tensors='pt')
target_data = df_train_single['complexity']




from torch.utils.data import Dataset

class WordcountDataset(Dataset):
    def __init__(self):
        pass
    
    def __len__(self):
        return len(df_train_single)
    
    def __getitem__(self, idx):
        input_ids = input_data['input_ids'][idx]
        token_type_ids = input_data['token_type_ids'][idx]
        attention_masks = input_data['attention_mask'][idx]
        out = target_data[idx]
        
        
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


# In[98]:


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
EPOCHS = 10
BATCH_SIZE = 8
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
        loss, predictions = train(bert,input_ids, attention_mask, token_ids, out, optm, criterion)
        epoch_loss+=loss
        #print("Predict time: {}".format(time.time() - start))
        
    print('Epoch {} Loss : {}'.format((epoch+1),epoch_loss))
