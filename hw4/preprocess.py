#encoding=utf-8

import jieba
import csv
import pickle
import numpy as np
import re
import os,sys
import gensim

dictionary = sys.argv[1]
data_name = sys.argv[2]
save_name = sys.argv[3]

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
data_jieba = jiebacut(data_name)
data = wordfilter(data_jieba,stopWords)

with open(save_name,"wb") as f:
    pickle.dump(data,f)
