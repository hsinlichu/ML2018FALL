#encoding=utf-8

import jieba
import csv
import pickle
import os.path
import numpy as np

dictionary = "dict.txt.big"
dataset_x = "train_x.csv"
word2ix_name = "word2ix.plk"
ix2cnt_name = "ix2cnt.plk"
embedding_name = "BOW_embedding.npy"
stopWords_name = "stopWords.txt"


jieba.set_dictionary(dictionary)

def jiebacut(dataset_x):
    cut_x = []
    with open(dataset_x,"r") as fin:
        raw_x = list(csv.reader(fin))[1:][:]
        for r in raw_x:
            sentance = "".join(r[1:])
            cut = jieba.cut(sentance,cut_all=False)
            cut_x.append(cut)
    print("number of sentance",len(cut_x))
    return cut_x
    
def readstopword(stopWords_name):
    stopWords = []
    with open(stopWords_name, 'r', encoding='UTF-8') as file:
        for data in file.readlines():
            data = data.strip()
            stopWords.append(data)
    return stopWords


word2ix = None
ix2cnt = None
if os.path.isfile(word2ix_name) and os.path.isfile(ix2cnt_name) and False:
    with open(word2ix_name, 'rb') as handle:
        word2ix = pickle.load(handle)
    with open(ix2cnt_name, 'rb') as handle:
        ix2cnt = pickle.load(handle)
else:
    stopWords = readstopword(stopWords_name)
    cut_x = jiebacut(dataset_x)
    word2ix = {}
    ix2cnt = [0] # the 0-index means other
    n = 1
    swcnt = 0
    for s in cut_x:
        for w in s:
            if w in stopWords or w == '\n':
                swcnt += 1
                continue
            if w in word2ix:
                ix2cnt[word2ix[w]] += 1
            else:
                word2ix[w] = n
                ix2cnt.append(1)
                n += 1

    with open(word2ix_name, 'wb') as handle:
        pickle.dump(word2ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ix2cnt_name, 'wb') as handle:
        pickle.dump(ix2cnt, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("{} Stopwords".format(swcnt))

len_word2ix = len(word2ix) + 1 # the addition one is for word not in word2ix
print(word2ix)
print("word2ix length",len_word2ix)
hjk

cut_x = jiebacut(dataset_x)
num_data = len(cut_x)
train_x = np.zeros((num_data,len_word2ix))
for i,s in enumerate(cut_x):
    for w in s:
        if w in stopWords or w == '\n':
            continue
        if w in word2ix:
            train_x[i][word2ix[w]] += 1
        else:
            train_x[i][0] += 1 # for word not in word2ix
print(train_x.shape)
np.save(embedding_name,train_x)
print("Embedding vector save as: {}".format(embedding_name))


