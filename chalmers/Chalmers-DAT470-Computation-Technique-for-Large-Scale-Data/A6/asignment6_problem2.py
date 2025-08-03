#!/usr/bin/env python3

import numpy as np
import pandas as pd
import csv
import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import time

def load_glove(filename):
    """
    Loads the glove dataset. Returns three things:
    A dictionary that contains a map from words to rows in the dataset.
    A reverse dictionary that maps rows to words.
    The embeddings dataset as a NumPy array.
    """
    df = pd.read_table(filename, sep=' ', index_col=0, header=None,
                           quoting=csv.QUOTE_NONE)
    word_to_idx = dict()
    idx_to_word = dict()
    for (i,word) in enumerate(df.index):
        word_to_idx[word] = i
        idx_to_word[i] = word
    return (word_to_idx, idx_to_word, df.to_numpy())

def normalize(X):
    """
    Reads an n*d matrix and normalizes all rows to have unit-length (L2 norm)
    
    Implement this function using array operations! No loops allowed.
    """
    data = np.array(X)
    data_normalized = (data-data.min())/(data.max()-data.min())
    column_size = len(data_normalized[0])
    row_size = len(data_normalized)
    # print(f"The rows are {row_size} and The column value for X is {column_size}")
    print(f"data_normalized is {data_normalized}")
    return data_normalized

def construct_queries(queries_fn, word_to_idx, X):
    """
    Reads queries (one string per line) and returns:
    - The query vectors as a matrix Q (one query per row)
    - Query labels as a list of strings
    """
    with open(queries_fn, 'r') as f:
        queries = f.read().splitlines()
    Q = np.zeros((len(queries), X.shape[1]))
    print(f"This is Q :{Q} before assigning X values")
    for i in range(len(queries)):
        Q[i,:] = X[word_to_idx[queries[i]],:]
    print(f"Q is of length:{len(Q)} and column size is {len(Q[0])}")
    return (Q,queries)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Glove dataset filename',
                            type=str)
    parser.add_argument('queries', help='Queries filename', type=str)
    args = parser.parse_args()
    
    (word_to_idx, idx_to_word, X) = load_glove(args.dataset)


    X = normalize(X)

    (Q,queries) = construct_queries(args.queries, word_to_idx, X)

    t1 = time.time()
    dot_product = np.dot(Q,X.transpose())   
    t2 = time.time()
    magnitude_X = np.linalg.norm(X)
    magnitude_Q = np.linalg.norm(Q)
    cosine_similarity = dot_product / (magnitude_Q * magnitude_X) 
    print(f"cosine_similarity is {cosine_similarity}")
    print('matrix multiplication took', t2-t1)
    

    # Compute here I such that I[i,:] contains the indices of the nearest
    # neighbors of the word i in ascending order.
    # Naturally, I[i,-1] should then be the index of the word itself.
    # raise NotImplementedError()
    X_train, X_test, y_train, y_test =train_test_split(X,Q,test_size=0.2,random_state=42)
    knn = KNeighborsClassifier(n_jobs=3)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    t3 = time.time()
    
    for i in range(I.shape[0]):
        neighbors = [idx_to_word[i] for i in I[i,-2:-5:-1]]
        print(f'{queries[i]}: {" ".join(neighbors)}')

    print('matrix multiplication took', t2-t1)
    print('sorting took', t3-t2)
    print('total time', t3-t1)
