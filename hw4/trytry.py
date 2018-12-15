#encoding=utf-8
import gensim
from gensim.models import Word2Vec
import keras
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Conv1D, Flatten, Dropout, Dense, Activation ,GRU,LeakyReLU
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
import os
import os.path
import numpy as np
import time

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

def get_session(gpu_fraction=0.8):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
ktf.set_session(get_session())



feature = "GRU_iter50_no_sw_2thgru512"
graphdir = "log2"

if not os.path.exists(graphdir):
    os.makedirs(graphdir)
if not os.path.exists(feature):
    os.makedirs(feature)

rnn_model_name = os.path.join(feature,'rnn.h5')
token_name = os.path.join(feature,'tokenizer.pkl')
wvmodel_name =os.path.join(feature,"w2vmodel.bin")
logdir = os.path.join(graphdir,feature)
processed_data = "no_stopword"

train_y_name = "train_y.csv"

EMBEDDING_DIM = 256
BATCH_SIZE = 512
MAX_LENGTH = 64

if not os.path.exists(logdir):
    os.makedirs(logdir)

raw_x = []
test_x = []
with open(os.path.join(processed_data,"train_x_processed.plk"),"rb") as f:
    raw_x = pickle.load(f)
with open(os.path.join(processed_data,"test_x_processed.plk"),"rb") as f:
    test_x = pickle.load(f)
   
raw_y = None
with open(train_y_name,"r") as fin:
    raw_y = list(csv.reader(fin))[1:][:]
    raw_y = [int(r[1]) for r in raw_y]

print("raw_x",len(raw_x))
print("raw_y",len(raw_y))
print("test_x",len(test_x))


whole = raw_x + test_x

print(raw_x[0])
wvmodel = Word2Vec(whole,size=EMBEDDING_DIM, workers=8,iter=50)
print("Word2Vec",wvmodel)
words = list(wvmodel.wv.vocab)
print("words len",len(words))


# save model
wvmodel.save(wvmodel_name)



raw_x = [" ".join(s) for s in raw_x]
test_x = [" ".join(s) for s in test_x]
 
t = Tokenizer()
t.fit_on_texts(raw_x)
vocab_size = len(t.word_index) + 1
print("vocab_size",vocab_size)


with open(token_name, 'wb') as handle:
    pickle.dump(t, handle)

raw_x = t.texts_to_sequences(raw_x)
raw_x = pad_sequences(raw_x,maxlen=MAX_LENGTH,padding='post')

train_x = []
train_y = []
val_x = []
val_y = []
for i in range(len(raw_x)):
    if i % 10 == 0:
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
for word, i in t.word_index.items():
    word2idx[word] = i
    if word in wvmodel :
        embedding_vector = wvmodel[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    else: 
        embedding_matrix[i] = embedding_matrix[0]

meta_file = "w2v_metadata.tsv"
word2idx_sorted = [(k, word2idx[k]) for k in sorted(word2idx, key = word2idx.get, reverse = False)]
with open(os.path.join(logdir, meta_file), 'w+') as file_metadata:
    for word in word2idx_sorted:
        if word[0] == '':
            print("Emply Line, should replecaed by any thing else, or will cause a bug of tensorboard")
            file_metadata.write('<Empty Line>' + '\n')
        else:
            file_metadata.write(word[0] + '\n') # save model wvmodel.save(wvmodel_name) 
embedding_layer = Embedding(vocab_size,EMBEDDING_DIM,weights=[embedding_matrix],input_length=MAX_LENGTH,trainable=False)

rnn = Sequential()
rnn.add(embedding_layer)
rnn.add(GRU(256, recurrent_dropout = 0.3, dropout=0.3, return_sequences=True))
rnn.add(BatchNormalization())
rnn.add(GRU(512, recurrent_dropout = 0.4, dropout=0.4))
rnn.add(BatchNormalization())

rnn.add(Dense(units=256,kernel_constraint=max_norm(5), kernel_initializer='glorot_normal'))
rnn.add(BatchNormalization())
rnn.add(LeakyReLU(alpha=1./20))
rnn.add(Dropout(0.5))
rnn.add(Dense(units=256,kernel_constraint=max_norm(5), kernel_initializer='glorot_normal'))
rnn.add(BatchNormalization())
rnn.add(LeakyReLU(alpha=1./20))
rnn.add(Dropout(0.5))
rnn.add(Dense(1,activation='sigmoid'))
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,clipvalue=0.5) #initial:lr=0.001

rnn.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
rnn.summary()

checkpoint = ModelCheckpoint(rnn_model_name, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
tbcallBack = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=True,embeddings_freq = 1,embeddings_layer_names = None,embeddings_metadata = meta_file)
#early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=0, mode='auto')
duration = -time.time()
rnn.fit(train_x, train_y, validation_data=(val_x,val_y),callbacks = [checkpoint,tbcallBack],epochs=50, batch_size=BATCH_SIZE, verbose=1)
duration += time.time()
print("Training time:{}(s)".format(duration))

print("Save in {}".format(feature))
print("EMBEDDING_DIM {} | BATCH_SIZE {} | MAX_LENGTH {}".format(EMBEDDING_DIM,BATCH_SIZE,MAX_LENGTH))

