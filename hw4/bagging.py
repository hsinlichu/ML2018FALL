#encoding=utf-8
import gensim
from gensim.models import Word2Vec
import keras
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Conv1D, Flatten, Dropout, Dense, Activation ,GRU
from keras.layers import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adam, Adadelta
from keras.models import Sequential
from keras.models import load_model
from keras.utils import np_utils

import csv
import pickle
import os.path
import numpy as np
import sys


EMBEDDING_DIM = 256
BATCH_SIZE = 512
MAX_LENGTH = 64

modellist = [0,3,7,9]

save_name = sys.argv[1]
processed_data = "model"
token_name = os.path.join(processed_data,'tokenizer.pkl')
wvmodel_name =os.path.join(processed_data,"w2vmodel.bin")



test_x = []
with open("test_x_processed.plk","rb") as f:
    test_x = pickle.load(f)
test_x = [" ".join(s) for s in test_x]
 
model = Word2Vec.load(wvmodel_name)

#t = Tokenizer()
t = pickle.load(open(token_name,'rb'))
vocab_size = len(t.word_index) + 1
print ("vocab_size",vocab_size)
print ("test_x length",len(test_x))

test_x = t.texts_to_sequences(test_x)
test_x = pad_sequences(test_x,maxlen=MAX_LENGTH,padding='post')
test_x = np.array(test_x)

y_test = np.zeros((len(test_x),1))
for i in modellist:
    feature = "model" + str(i)
    rnn_model_name = os.path.join(feature,'rnn.h5')
    print(rnn_model_name)
    predict_model = load_model(rnn_model_name)
    y_test += predict_model.predict(test_x,batch_size=BATCH_SIZE, verbose=1)
y_test /= len(modellist)

with open(save_name,'w') as fout:
    writer = csv.writer(fout)
    writer.writerow(["id","label"])
    for i in range(len(y_test)):
        if(y_test[i] >= 0.5):
            writer.writerow([i,str(1)])
        else : 
            writer.writerow([i,str(0)])
