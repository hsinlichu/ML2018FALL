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
BATCH_SIZE = 256
MAX_LENGTH = 64
feature = sys.argv[2]

rnn_model_name = os.path.join(feature,'rnn.h5')
token_name = os.path.join(feature,'tokenizer.pkl')
wvmodel_name =os.path.join(feature,"w2vmodel.bin")
processed_data = "no_stopword"



test_x = []
with open(os.path.join(processed_data,"test_x_processed.plk"),"rb") as f:
    test_x = pickle.load(f)
test_x = [" ".join(s) for s in test_x]
 
model = Word2Vec.load(wvmodel_name)

#t = Tokenizer()
t = pickle.load(open(token_name,'rb'))
vocab_size = len(t.word_index) + 1
print (vocab_size)
print (len(test_x))

test_x = t.texts_to_sequences(test_x)
test_x = pad_sequences(test_x,maxlen=MAX_LENGTH,padding='post')
test_x = np.array(test_x)

predict_model = load_model(rnn_model_name)
predict_model.summary()
y_test = predict_model.predict(test_x,batch_size=BATCH_SIZE, verbose=1)

with open(sys.argv[1],'w') as fout:
    writer = csv.writer(fout)
    writer.writerow(["id","label"])
    for i in range(len(y_test)):
        if(y_test[i] >= 0.5):
            writer.writerow([i,str(1)])
        else : 
            writer.writerow([i,str(0)])
