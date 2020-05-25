import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
# from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertModel
# from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import os
import torch.nn.functional as F

from types import SimpleNamespace
import pudb
import sys
# from bert_serving.client import BertClient
# bc = BertClient()
from sklearn.feature_extraction.text import TfidfVectorizer

try:
	dataset = sys.argv[1]
	if dataset not in ['nepal','italy']:
		dataset='nepal'
except Exception as e:
		dataset='nepal'

with open('DATA_2/INPUT/nepal_dict.p','rb') as handle:
	nepal_dict= pickle.load( handle)

with open('DATA_2/INPUT/italy_dict.p','rb') as handle:
	italy_dict= pickle.load(handle)


nepal_dict ={}

if dataset =='nepal':
	need_file = open('DATA_2/INPUT/nepal_needs.txt', encoding="utf-8")
	offer_file = open('DATA_2/INPUT/nepal_offers.txt', encoding="utf-8")
	all_file =open('DATA_2/INPUT/nepal-all.txt', encoding="utf-8")
else:
	need_file = open('./DATA_2/INPUT/italy_needs.txt', encoding="utf-8")
	offer_file = open('./DATA_2/INPUT/italy_offers.txt', encoding="utf-8")
	all_file =open('./DATA_2/INPUT/italy-all.txt', encoding="utf-8")

while(True):
	line = need_file.readline()
	if not line: break
	line=line.strip().split('<||>')
	nepal_dict[line[0]]=(line[1].lower(), 1)

while(True):
	line = offer_file.readline()
	if not line: break
	line=line.strip().split('<||>')
	nepal_dict[line[0]]=(line[1].lower(), 2)

while(True):
	line = all_file.readline()
	if not line: break
	line= line.strip().split('<||>')
	if line[0] not in nepal_dict:
		nepal_dict[line[0]]= (line[1].lower(),0)

print(len(nepal_dict))

def create_train_test_data(nepal_dict):
	X=[[],[],[]]
	
	for elem in nepal_dict:
		X[nepal_dict[elem][1]].append(nepal_dict[elem][0])
	
	random.shuffle(X[0])
	random.shuffle(X[1])
	random.shuffle(X[2])

	train = [(X[i][k],i) for i in range(0,3) for k in range(0,int(0.7*len(X[i]))) ]
	val   = [(X[i][k],i) for i in range(0,3) for k in range(int(0.7*len(X[i])),int(0.8*len(X[i])))] 
	test  = [(X[i][k],i) for i in range(0,3) for k in range(int(0.8*len(X[i])), len(X[i]))]
	
	random.shuffle(train)
	random.shuffle(val)
	random.shuffle(test)  
	
	return train, val,  test

if dataset=='nepal':
	train_nepal, val_nepal, test_nepal = create_train_test_data(nepal_dict)
else:
	train_nepal, val_nepal, test_nepal = create_train_test_data(italy_dict)

# Reduce size of data
# train_nepal = train_nepal[:10]
# val_nepal = val_nepal[:10]
# test_nepal = test_nepal[:10]

train_nepal_sentences = ["[CLS] "+ text[0]+ " [SEP]" for text in train_nepal]
val_nepal_sentences   = ["[CLS] "+ text[0]+ " [SEP]" for text in val_nepal]
test_nepal_sentences  = ["[CLS] "+ text[0]+ " [SEP]" for text in test_nepal]
# train_italy_sentences = ["[CLS] "+ text[0]+ " [SEP]" for text in train_italy]
# val_italy_sentences   = ["[CLS] "+ text[0]+ " [SEP]" for text in val_italy]
# test_italy_sentences  = ["[CLS] "+ text[0]+ " [SEP]" for text in test_italy]

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# tokenized_nepal_train = [tokenizer.tokenize(sent) for sent in train_nepal_sentences]
# tokenized_nepal_val   = [tokenizer.tokenize(sent) for sent in val_nepal_sentences]
# tokenized_nepal_test  = [tokenizer.tokenize(sent) for sent in test_nepal_sentences]

# tokenized_italy_train = [tokenizer.tokenize(sent) for sent in train_italy_sentences]
# tokenized_italy_val   = [tokenizer.tokenize(sent) for sent in val_nepal_sentences]
# tokenized_italy_test  = [tokenizer.tokenize(sent) for sent in test_italy_sentences]

train_nepal_labels    = [elem[1] for elem in train_nepal]
val_nepal_labels      = [elem[1] for elem in val_nepal]
test_nepal_labels     = [elem[1] for elem in test_nepal]
# train_italy_labels    = [elem[1] for elem in train_italy]
# val_italy_labels      = [elem[1] for elem in val_italy]
# test_italy_labels     = [elem[1] for elem in test_italy]

f = open("DATA_2/INPUT/ft/train_nepal", "w")

for i, label in enumerate(train_nepal_labels + val_nepal_labels):
	line = "__label__"+str(label)
	line = line + " " + (train_nepal_sentences + val_nepal_sentences)[i]
	f.write(line+"\n")
os.system("cat DATA_2/INPUT/ft/train_nepal | sed -e \"s/\([.\!?,'/()]\)/ \1 /g\" | tr \"[:upper:]\" \"[:lower:]\" > DATA_2/INPUT/ft/train_nepal_processed")

f.close()

# f = open("DATA_2/INPUT/ft/val_nepal", "w")

# for i, label in enumerate(val_nepal_labels):
# 	line = "__label__"+str(label)
# 	line = line + " " + val_nepal_sentences[i]
# 	f.write(line+"\n")
# os.system("cat DATA_2/INPUT/ft/val_nepal | sed -e \"s/\([.\!?,'/()]\)/ \1 /g\" | tr \"[:upper:]\" \"[:lower:]\" > DATA_2/INPUT/ft/val_nepal_processed")


# f.close()

f = open("DATA_2/INPUT/ft/test_nepal", "w")

for i, label in enumerate(test_nepal_labels):
	line = "__label__"+str(label)
	line = line + " " + test_nepal_sentences[i]
	f.write(line+"\n")
os.system("cat DATA_2/INPUT/ft/test_nepal | sed -e \"s/\([.\!?,'/()]\)/ \1 /g\" | tr \"[:upper:]\" \"[:lower:]\" > DATA_2/INPUT/ft/test_nepal_processed")

f.close()

print("Done")