#!/usr/bin/env python
# coding: utf-8

# ### Dependencies ###

# In[96]:


import os
import torch
import numpy as np
import time
import pandas as pd

from scipy import spatial

# ### Train data ###

# In[97]:


train_single_tsv = '../dataset/train/lcp_single_train.tsv'
df_train_single = pd.read_csv(train_single_tsv, sep='\t', header=0)


# In[98]:


print("Data columns: \n")
print(df_train_single.columns)
print("Total corpus len: {}".format(len(df_train_single)))
print("Subcorpus len:\n")
print(df_train_single['corpus'].value_counts())


# ### Test data 

# In[101]:


test_single_tsv = '../dataset/test/lcp_single_test.tsv'
df_test_single = pd.read_csv(test_single_tsv, sep='\t', header=0)


# In[102]:


print("Data columns: \n")
print(df_test_single.columns)
print("Total corpus len: {}".format(len(df_test_single)))
print("Subcorpus len:\n")
print(df_test_single['corpus'].value_counts())
print(os.getcwd())


# ### Trial data 
# 

# In[103]:


trial_single_tsv = '../dataset/trial/lcp_single_trial.tsv'
df_trial_single = pd.read_csv(trial_single_tsv, sep='\t', header=0)


# In[104]:


print("Data columns: \n")
print(df_trial_single.columns)
print("Total corpus len: {}".format(len(df_trial_single)))
print("Subcorpus len:\n")
print(df_trial_single['subcorpus'].value_counts())
print(os.getcwd())


# ### GloVe ###
# Load the pretrained GloVe vectors and verify that the operation has been successful by some quick experiments with the embedding.  

# In[105]:


glove_w2v_loc = '../InferSent/GloVe/glove.6B.300d.txt'
with open(glove_w2v_loc,  "r", encoding="utf8") as lines:
    glove_w2v = {}
    for line in lines:
        values = line.split()
        word = ''.join(values[:-300])
        vector = np.asarray(values[-300:], dtype='float32')
        glove_w2v[word.lower()] = vector
    print(len(glove_w2v)," words loaded!")


# In[106]:


def find_closest_embeddings(embedding):
    return sorted(glove_w2v.keys(), key=lambda word: spatial.distance.euclidean(glove_w2v[word.lower()], embedding))[0:5]

# ### InferSent
# - https://towardsdatascience.com/learning-sentence-embeddings-by-natural-language-inference-a50b4661a0b8
# - https://research.fb.com/downloads/infersent/
# 
# Load InferSent model and execute some experiments.  
# Currently it is using GloVe. We have tested GloVe and fastText and have chosen GloVe over fastText.

# In[109]:


from InferSent.models import InferSent


# In[110]:


model_pkl = '../InferSent/encoder/infersent1.pkl'
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
infer_sent_model = InferSent(params_model)
infer_sent_model.load_state_dict(torch.load(model_pkl))


# In[111]:


infer_sent_model.set_w2v_path(glove_w2v_loc)
infer_sent_model.build_vocab_k_words(K=100000)

# infer_sent_model.to(torch.device("cuda:0"))


# In[112]:


infer_sent_model.encode(["This man is playing computer games"], tokenize=True)


# In[113]:


def get_embedding_for_context(ctx):
    if not isinstance(ctx, list):
#       print("ctx is not list")
        ctx = [ctx]
    return infer_sent_model.encode(ctx, tokenize=True)

# In[114]:


from sklearn.metrics.pairwise import cosine_similarity

def measure_dist_between_ctx(c1, c2):
    e1 = get_embedding_for_context(c1)[0]
    e2 = get_embedding_for_context(c2)[0]
    #return spatial.distance.euclidean(e1, e2)
    return cosine_similarity([e1], [e2])

# ### Handcrafted features
# 
# * Word length
# * Syllable count
# * Word frequency

# In[115]:


import syllables
# According to the paper there are 3 handcrafted features
# - word length
# - word frequency 
# - syllable count
import csv
import math
import nltk

from collections import defaultdict
from nltk.stem.porter import PorterStemmer

reader = csv.reader(open('SUBTLEX.csv', 'r'))
frequency = defaultdict(float)
frequency_count = dict()
stemmer = PorterStemmer()

for row in reader:
    token = stemmer.stem(row[0].lower())
    frequency[token] += float(row[5])

frequency = {k: math.log2(v) for k, v in frequency.items()}

def get_handcrafted_features(word):
    word = str(word)
    return [len(word), syllables.estimate(word), frequency.get(stemmer.stem(word.lower())) or 0]

get_handcrafted_features("Basketball")


# ### Load datasets

# In[116]:


from torch.utils.data import Dataset

def preprocess_embeddings(dataset):
    # Preprocess all sentence embeddings for the data:
    sentence_embeddings = {}
    
    all_sentences = dataset['sentence'].tolist()

    start = time.time()
    all_sentence_embeddings = get_embedding_for_context(all_sentences)
    print("Encoding time for all sentences: {}".format(time.time() - start))
    return all_sentence_embeddings
    

class CompLexDataset(Dataset):
    global dataset_type
    
    def __init__(self, dataset_type):
        self.dataset_type = dataset_type
        
        if(self.dataset_type == 'train'):                   
            self.all_sentence_embeddings = preprocess_embeddings(df_train_single)
        elif(self.dataset_type == 'trial'):
            self.all_sentence_embeddings = preprocess_embeddings(df_trial_single)
        elif(self.dataset_type == 'test'):
            self.all_sentence_embeddings = preprocess_embeddings(df_test_single)
    
    def __len__(self):
        if(self.dataset_type == 'train'):                   
            return len(df_train_single)
        elif(self.dataset_type == 'trial'):
            return len(df_trial_single)
        elif(self.dataset_type == 'test'):
            return len(df_test_single)
        else: 
            raise Exception("Invalid dataset type.", self.dataset_type)

    
    def __getitem__(self,idx):
        start = time.time()
        if(self.dataset_type == 'train'):
            token = df_train_single.loc[idx, 'token']
            token = str(token)
            out = df_train_single.loc[idx, 'complexity']
        elif(self.dataset_type == 'trial'):
            token = df_trial_single.loc[idx, 'token']
            token = str(token)
            out = df_trial_single.loc[idx, 'complexity']
        elif(self.dataset_type == 'test'):
            token = df_test_single.loc[idx, 'token']
            token = str(token)
            out = df_test_single.loc[idx, 'complexity']
        else: 
            raise Exception("Invalid dataset type.", self.dataset_type)
        
        handcrafted_features = get_handcrafted_features(token)

        sentence_ctx = self.all_sentence_embeddings[idx]
        
        if token.lower() in glove_w2v:   
            w2v_for_token = glove_w2v[token.lower()]
        else:
            #print("Token {} not found".format(token.lower()))
            w2v_for_token = [0] * 300
        
        
        result = {
            'inp': torch.from_numpy(np.hstack((np.array(handcrafted_features), sentence_ctx, np.array(w2v_for_token))).ravel()).float(), 
            'out': torch.from_numpy(np.array([out])).float()
        }
        
        #print("Idx {} fetch time: {}".format(idx, time.time() - start))
        return result
    


# In[80]:


train_dataset = CompLexDataset("train")
print("Input: ", train_dataset[5]['inp'], "Input Length: ", len(train_dataset[5]['inp']))
print("Output: ", train_dataset[5]['out'])

trial_dataset = CompLexDataset("trial")
print("Input: ", trial_dataset[5]['inp'], "Input Length: ", len(trial_dataset[5]['inp']))
print("Output: ", trial_dataset[5]['out'])

test_dataset = CompLexDataset("test")
print("Input: ", test_dataset[5]['inp'], "Input Length: ", len(test_dataset[5]['inp']))
print("Output: ", test_dataset[5]['out'])


# ### Network ###

# In[118]:


import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(len(train_dataset[0]['inp']), 1600)
        self.b1 = nn.BatchNorm1d(1600)
        self.fc2 = nn.Linear(1600, 1)
        self.sigmoid = nn.Sigmoid() 


    def forward(self,x):

        x = self.fc1(x)
        #x = self.b1(x)
        x = self.fc2(x)

        #return x
        return self.sigmoid(x)
        
    
net = Network()
print(net)
#net.to(torch.device("cuda:0"))
train_dataset[0]


# In[119]:


def train(model, x, y, optimizer, criterion):
    model.zero_grad()
    output = model(x)
    loss = criterion(output,y)
    loss.backward()
    optimizer.step()

    return loss, output


# In[120]:


#print(torch.cuda.is_available())
#print(torch.cuda.current_device())


# ### Mean Squared Error ###
# Training phase

# In[121]:


from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

criterion = nn.MSELoss()
EPOCHS = 30
BATCH_SIZE = 64
optm = Adam(net.parameters(), lr = 0.00001)

data_train = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)

for epoch in range(EPOCHS):
    epoch_loss = 0
    correct = 0
    
    for bidx, batch in enumerate(data_train):
        #start = time.time()
        x_train = batch['inp']
        y_train = batch['out']
        #print("Fetch time: {}".format(time.time() - start))
        
        #start = time.time()
        loss, predictions = train(net,x_train,y_train, optm, criterion)
        epoch_loss += loss
        #print("Predict time: {}".format(time.time() - start))
        
    print('Epoch {} Loss : {}'.format((epoch+1),epoch_loss))


# ### Output for single sample

# In[122]:


net(train_dataset[210]['inp'])


# ### Mean Absolute Error ###

# #### MAE for test dataset

# In[132]:


from sklearn.metrics import mean_absolute_error

y_true = [test_dataset[i]['out'].item() for i in range(len(test_dataset))]
y_pred = []

test_loader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False)
for bidx, batch in enumerate(test_loader):
        #start = time.time()
        x_train = batch['inp']
        y_pred.append(net(x_train))

y_pred = [x.item() for i in range(len(y_pred)) for x in y_pred[i] ]

mae = mean_absolute_error(y_true, y_pred)
print("MAE for test data: ", mae)

with open('test_results.csv', 'w', newline='') as f:
    f_writer = csv.writer(f, delimiter=',',)
    for idx in range(len(df_test_single)):
       f_writer.writerow((df_test_single.loc[idx, 'id'], str(y_pred[idx])))


# #### MAE for trial dataset

# In[124]:


from sklearn.metrics import mean_absolute_error

y_true = [trial_dataset[i]['out'].item() for i in range(len(trial_dataset))]
y_pred = []

trial_loader = DataLoader(dataset = trial_dataset, batch_size = BATCH_SIZE, shuffle = False)
for bidx, batch in enumerate(trial_loader):
        #start = time.time()
        x_train = batch['inp']
        y_pred.append(net(x_train))

y_pred = [x.item() for i in range(len(y_pred)) for x in y_pred[i] ]

mae = mean_absolute_error(y_true, y_pred)
print("MAE for trial data: ", mae)


# #### MAE for train dataset

# In[125]:


from sklearn.metrics import mean_absolute_error

y_true = [train_dataset[i]['out'].item() for i in range(len(train_dataset))]
y_pred = []

test_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = False)
for bidx, batch in enumerate(test_loader):
        #start = time.time()
        x_train = batch['inp']
        y_pred.append(net(x_train))

y_pred = [x.item() for i in range(len(y_pred)) for x in y_pred[i] ]

mae = mean_absolute_error(y_true, y_pred)
print("MAE for train data: ", mae)


# #### MAE for total random

# In[126]:


from sklearn.metrics import mean_absolute_error
import random

y_true = [train_dataset[i]['out'].item() for i in range(len(train_dataset))]
y_pred = [random.random() for i in range(len(train_dataset))]


mae = mean_absolute_error(y_true, y_pred)
print("Mean Absolute Error for train data: ", mae)


# #### Demo

# In[95]:


def prepare_sentence(sentence, token):
    sentence_embeddings = get_embedding_for_context(sentence)[0]
    handcrafted_features = get_handcrafted_features(token)
            
    if token.lower() in glove_w2v:   
        w2v_for_token = glove_w2v[token.lower()]
    else:
       w2v_for_token = [0] * 300
    
    return {
            'inp': torch.from_numpy(np.hstack((np.array(handcrafted_features), sentence_embeddings, np.array(w2v_for_token))).ravel()).float() 
           }

    


# In[128]:

# In[1]:





# In[ ]:




