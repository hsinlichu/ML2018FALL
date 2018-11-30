import csv
import time
import sys,os
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers.core import Activation
from keras.constraints import max_norm
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator

def read_training_data(filename,seed):
    train_x, train_y = [],[]
    val_x,val_y = [],[]
    with open(filename,"r") as fin:
        tmp = list(csv.reader(fin))[1:]
        np.random.shuffle(tmp)
        for index in range(len(tmp)):
           label,image = tmp[index][0],tmp[index][1]
           pixel = np.array([int(i) for i in image.split(" ")]).reshape(48,48,1)
           if index % 10 == seed:
               val_x.append(pixel)
               val_y.append(label)
           else:
               train_x.append(pixel)
               train_y.append(label)

    train_x = np.array(train_x, dtype=float) / 255
    train_y = np.array(train_y, dtype=int)
    val_x = np.array(val_x, dtype=float) / 255
    val_y = np.array(val_y, dtype=int)
    train_y = np_utils.to_categorical(train_y,7)
    val_y = np_utils.to_categorical(val_y,7)

    print("train_x",train_x.shape)        
    print("train_y",train_y.shape)        
    print("val_x",val_x.shape)        
    print("val_y",val_y.shape)        
    
    return train_x,train_y,val_x,val_y

def cnnmodel(lr):
    afun = LeakyReLU(alpha=1./20) #Activation("relu")
    model = Sequential()
    model.add(Dropout(0.0,input_shape=(48,48,1))) # dropout on the inputs
    model.add(Conv2D(64,(3,3),padding="same", kernel_constraint=max_norm(5), kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=1./20))
    

    model.add(Conv2D(128,(3,3),padding='same', kernel_constraint=max_norm(5), kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=1./20))

    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(256,(3,3),padding='same', kernel_constraint=max_norm(5), kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=1./20))
    
    model.add(Conv2D(256,(3,3),padding='same', kernel_constraint=max_norm(5), kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=1./20))

    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(512,(3,3),padding='same', kernel_constraint=max_norm(5), kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=1./20))

    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(units=1024,kernel_constraint=max_norm(5), kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=1./20))
    model.add(Dropout(0.5))

    model.add(Dense(units=1024,kernel_constraint=max_norm(5), kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(units=7, activation='softmax',kernel_constraint=max_norm(5)))

    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

if __name__ == '__main__':
    BATCH_SIZE = 128 
    epoch = 400
    lr = 0.001
    traindata = sys.argv[1]
    seed = int(sys.argv[2])
    model_path = "cnn400.h5"
    print("Epoch {} | Batch size {} | Leraning rate {} | Model path {}".format(epoch,BATCH_SIZE,lr,model_path))

    
    train_x,train_y,val_x,val_y = read_training_data(traindata,seed)
    model = cnnmodel(lr)
    datagen = ImageDataGenerator(
        samplewise_center=False,samplewise_std_normalization=False,       
        featurewise_center=False,featurewise_std_normalization=False,
        rotation_range=15,width_shift_range=0.1,height_shift_range=0.1,
        horizontal_flip=True,zca_whitening=False)
    datagen.fit(train_x)

    duration = -time.time()
    tbCallBack = TensorBoard(log_dir='./cnnGraph', histogram_freq=0, write_graph=True, write_images=True)
    checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    model.fit_generator(datagen.flow(train_x,train_y,batch_size=BATCH_SIZE,shuffle=True),validation_data=(val_x,val_y),steps_per_epoch=len(train_x)//BATCH_SIZE,callbacks=[checkpoint,tbCallBack],epochs=epoch,workers = 10)
    duration += time.time()
    print("Training duration:{}(s)",duration)
    print("Epoch {} | Batch size {} | Leraning rate {} | Model path {}".format(epoch,BATCH_SIZE,lr,model_path))
    print("=======================================================================================")
