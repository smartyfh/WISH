import os
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
import pickle
from tqdm import tqdm
import argparse
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import csv
import gzip
from sklearn.datasets import fetch_20newsgroups

home = str(Path.home())

all_docs = []
all_tags = []
newsgroups_train = fetch_20newsgroups(subset='train')
for i in range(len(newsgroups_train.data)):
    all_docs.append(newsgroups_train.data[i])

for i in range(len(newsgroups_train.data)):
    all_tags.append(newsgroups_train.target[i])
    
newsgroups_test = fetch_20newsgroups(subset='test')
for i in range(len(newsgroups_test.data)):
    all_docs.append(newsgroups_test.data[i])
    
for i in range(len(newsgroups_test.data)):
    all_tags.append(newsgroups_test.target[i])
    
print(len(all_docs), len(all_tags))
print(all_tags[0:2], all_docs[0:2])


num = len(all_docs)
n_test = int(0.1 * num)
n_val = n_test
n_train = num - n_test * 2

train_docs = all_docs[:n_train]
val_docs = all_docs[n_train:num-n_test]
test_docs = all_docs[-n_test:]

train_tags = all_tags[:n_train]
val_tags = all_tags[n_train:num-n_test]
test_tags = all_tags[-n_test:]
print('num train:{} num test:{} num tags:{}'.format(len(train_tags), len(test_tags), len(set(train_tags))))

tfidf = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=3)

train_tf = tfidf.fit_transform(train_docs)
val_tf = tfidf.transform(val_docs)
test_tf = tfidf.transform(test_docs)

def create_dataframe(doc, target):
    docs = []
    for i, bow in enumerate(doc):
        d = {'doc_id': i, 'bow': bow, 'label': target[i]}
        docs.append(d)
    df = pd.DataFrame.from_dict(docs)
    df.set_index('doc_id', inplace=True)
    return df
    
train_df = create_dataframe(train_tf, train_tags)
val_df = create_dataframe(val_tf, val_tags)
test_df = create_dataframe(test_tf, test_tags)

save_dir = 'dataset/{}'.format('ng20')
train_df.to_pickle(os.path.join(save_dir, 'train.tf.df.pkl'))
test_df.to_pickle(os.path.join(save_dir, 'test.tf.df.pkl'))
val_df.to_pickle(os.path.join(save_dir, 'validation.tf.df.pkl'))
