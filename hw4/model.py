#encoding=utf-8
import gensim
from gensim.models import Word2Vec
import keras
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Conv1D, Flatten, Dropout, Dense, Activation ,GRU,LeakyReLU,Bidirectional
from keras.constraints import max_norm
from keras.layers import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping,TensorBoard,ModelCheckpoint
from keras.optimizers import SGD, Adam, Adadelta
from keras.models import Sequential
from keras.models import load_model
from keras.utils import np_utils

import csv
import pickle
import os,sys
import os.path
import numpy as np
import time

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

def get_session(gpu_fraction=0.8):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
ktf.set_session(get_session())


para = sys.argv[1]
feature = "checkpoint" + para
print("Save in {}".format(feature))

if not os.path.exists(feature):
    os.makedirs(feature)

processed_data = "."#"no_stopword_iter50"
rnn_model_name = os.path.join(feature,'rnn.h5')
token_name = os.path.join(processed_data,'tokenizer.pkl')
wvmodel_name =os.path.join(processed_data,"w2vmodel.bin")

train_y_name = sys.argv[2]

EMBEDDING_DIM = 256
BATCH_SIZE = 512
MAX_LENGTH = 64


raw_x = []
with open(os.path.join(processed_data,"train_x_processed.plk"),"rb") as f:
    raw_x = pickle.load(f)

raw_y = None
with open(train_y_name,"r") as fin:
    raw_y = list(csv.reader(fin))[1:][:]
    raw_y = [int(r[1]) for r in raw_y]

print("raw_x",len(raw_x))
print("raw_y",len(raw_y))
#print("test_x",len(test_x))


# load data
wvmodel = Word2Vec.load(wvmodel_name)
t = pickle.load(open(token_name,'rb'))
vocab_size = len(t.word_index) + 1
print("vocab_size",vocab_size)
raw_x = [" ".join(s) for s in raw_x]

raw_x = t.texts_to_sequences(raw_x)
raw_x = pad_sequences(raw_x,maxlen=MAX_LENGTH,padding='post')

train_x = []
train_y = []
val_x = []
val_y = []
for i in range(len(raw_x)):
    if i % 10 == int(para):
        val_x.append(raw_x[i])
        val_y.append(raw_y[i])
    else:
        train_x.append(raw_x[i])
        train_y.append(raw_y[i])

train_x = np.array(train_x)
train_y = np.array(train_y)
val_x = np.array(val_x)
val_y = np.array(val_y)
print("train_x",train_x.shape)
print("val_x",val_x.shape)

word2idx = {"PAD":0}
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
tmp = 0
for word, i in t.word_index.items():
    word2idx[word] = i
    if word in wvmodel:
        embedding_vector = wvmodel[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    else: 
        tmp += 1
        embedding_matrix[i] = embedding_matrix[0]

embedding_layer = Embedding(vocab_size,EMBEDDING_DIM,weights=[embedding_matrix],input_length=MAX_LENGTH,trainable=False)

rnn = Sequential()
rnn.add(embedding_layer)
rnn.add(GRU(256, recurrent_dropout = 0.5, dropout=0.5,return_sequences=True,kernel_initializer=keras.initializers.Orthogonal(gain=1.0, seed=None),bias_initializer='ones')) #keras.initializers.Orthogonal(gain=1.0, seed=None) #'glorot_normal'
rnn.add(BatchNormalization())
rnn.add(GRU(512, recurrent_dropout = 0.5, dropout=0.5,kernel_initializer=keras.initializers.Orthogonal(gain=1.0, seed=None),bias_initializer='ones'))
rnn.add(BatchNormalization())

rnn.add(Dense(units=256,kernel_constraint=max_norm(5), kernel_initializer='glorot_normal'))
rnn.add(BatchNormalization())
rnn.add(LeakyReLU(alpha=1./20))
rnn.add(Dropout(0.5))
rnn.add(Dense(units=128,kernel_constraint=max_norm(5), kernel_initializer='glorot_normal'))
rnn.add(BatchNormalization())
rnn.add(LeakyReLU(alpha=1./20))
rnn.add(Dropout(0.5))

rnn.add(Dense(units=64,kernel_constraint=max_norm(5), kernel_initializer='glorot_normal'))
rnn.add(BatchNormalization())
rnn.add(LeakyReLU(alpha=1./20))
rnn.add(Dropout(0.5))

rnn.add(Dense(1,activation='sigmoid'))
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,clipvalue=0.5) #initial:lr=0.001

rnn.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
rnn.summary()

checkpoint = ModelCheckpoint(rnn_model_name, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
duration = -time.time()
rnn.fit(train_x, train_y, validation_data=(val_x,val_y),callbacks = [checkpoint],epochs=50, batch_size=BATCH_SIZE, verbose=1)
duration += time.time()
print("Training time:{}(s)".format(duration))

print("Save in {}".format(feature))
print("EMBEDDING_DIM {} | BATCH_SIZE {} | MAX_LENGTH {}".format(EMBEDDING_DIM,BATCH_SIZE,MAX_LENGTH))

