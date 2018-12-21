#encoding=utf-8

import jieba
import csv
import pickle
import numpy as np
import re
import os
import gensim
from gensim.models import Word2Vec
import keras
from keras.preprocessing.text import Tokenizer

EMBEDDING_DIM = 256
BATCH_SIZE = 512
MAX_LENGTH = 64


feature = "precessed"
if not os.path.exists(feature):
    os.makedirs(feature)

dictionary = sys.argv[1]
train_x_name = sys.argv[2]
test_x_name = sys.argv[3]

wvmodel_name =os.path.join(feature,"w2vmodel.bin")
token_name = os.path.join(feature,'tokenizer.pkl')

jieba.set_dictionary(dictionary)

def remove_punctuation(line):
    rule = re.compile(r"[^a-zA-Z\u4e00-\u9fa5]")
    line = rule.sub('',line)
    return line

def jiebacut(dataset_x):
    cut_x = []
    with open(dataset_x,"r") as fin:
        raw_x = list(csv.reader(fin))[1:][:]
        for r in raw_x:
            sentance = "".join(r[1:])
            sentance = remove_punctuation(sentance)
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

def wordfilter(data,stopWords):
    maxlen = 0
    idx = None
    ret = []
    cnt = 0
    for i,s in enumerate(data):
        tmp = []
        for w in s:
            if w == "b" or w == "B":
                continue
            if w not in stopWords:
                tmp.append(w)
            else:
                cnt += 1
        ret.append(tmp)
        if len(tmp) > maxlen:
            maxlen = len(tmp)
            idx = i
        print("Has done {:04.2f}%...\r".format((i+1)/len(data)),end='')
    print("Sentence number:",len(ret))
    print("remove {} stopwords".format(cnt))
    return ret

stopWords = []
train_jieba = jiebacut(train_x_name)
test_jieba = jiebacut(test_x_name)
train_x = wordfilter(train_jieba,stopWords)
test_x = wordfilter(test_jieba,stopWords)

whole = train_x + test_x
wvmodel = Word2Vec(whole,size=EMBEDDING_DIM,min_count=5,window=5, workers=10,iter=20)
print("Word2Vec",wvmodel)
words = list(wvmodel.wv.vocab)
print("words len",len(words))

# save model
wvmodel.save(wvmodel_name)

t = Tokenizer()
t.fit_on_texts(whole)

with open(token_name, 'wb') as handle:
    pickle.dump(t, handle)

with open(os.path.join(feature,"train_x_processed.plk"),"wb") as f:
    pickle.dump(train_x,f)
with open(os.path.join(feature,"test_x_processed.plk"),"wb") as f:
    pickle.dump(test_x,f)
