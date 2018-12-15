#encoding=utf-8

import jieba
import csv
import pickle
import numpy as np
import re
import os

dictionary = "dict.txt.big"
train_x_name = "train_x.csv"
test_x_name = "test_x.csv"
stopWords_name = "stopWords.txt"
feature = "no_stopword"


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

def wordfilter(data):
    maxlen = 0
    idx = None
    ret = []
    for i,s in enumerate(data):
        tmp = []
        for w in s:
            if w == "b" or w == "B":
                continue
            if w not in stopWords:
                tmp.append(w)
        ret.append(tmp)
        if len(tmp) > maxlen:
            maxlen = len(tmp)
            idx = i
        print("Has done {:04.2f}%...\r".format((i+1)/len(data)),end='')
    print("Sentence number:",len(ret))
    print("max len",maxlen)
    print(idx)
    return ret

stopWords = readstopword(stopWords_name)
stopWords = []
train_jieba = jiebacut(train_x_name)
test_jieba = jiebacut(test_x_name)
train_x = wordfilter(train_jieba)
test_x = wordfilter(test_jieba)

if not os.path.exists(feature):
    os.makedirs(feature)

with open(os.path.join(feature,"train_x_processed.plk"),"wb") as f:
    pickle.dump(train_x,f)
with open(os.path.join(feature,"test_x_processed.plk"),"wb") as f:
    pickle.dump(test_x,f)

