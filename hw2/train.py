import csv
import numpy as np
from numpy.linalg import inv
import sys,os

def readtrainingdata(train_x,train_y):
    x = None
    y = None
    with open(train_x,"r") as fin:
        x = list(csv.reader(fin))[1:]
        x = [[int(n) for n in r] for r in x]
        print("origin x data:",len(x))
    with open(train_y,"r") as fin:
        y = list(csv.reader(fin))[1:]
        y = [int(n[0]) for n in y]
        print("origin y data:",len(y))
    x0 = []
    x1 = []
    for f,l in zip(x,y):
        if l == 1:
            x1.append(f)
        else:
            x0.append(f)
    x0 = np.array(x0).T
    x1 = np.array(x1).T
    print("x0",x0.shape)
    print("x1",x1.shape)
    return np.array(x),np.array(y),x0,x1

def predict(w,b,test_x,output):
    test = None
    with open(test_x,"r") as fin:
        test = list(csv.reader(fin))[1:]
        test = [[int(n) for n in r] for r in test]
    predict = sigmoid(test@w + b)
    predict = predict < 0.5 # mean the probability of belonging to 1
    #print(predict)
    out = [['id','value']]
    for index,v in enumerate(predict):
        if v:
            out.append(['id_'+str(index),1])
        else:
            out.append(['id_'+str(index),0])


    with open(output,"w") as fout:
        writer = csv.writer(fout)
        writer.writerows(out)	

def ProbabilisticGenerativeModel(x0,x1):
    n0 = x0.shape[1]
    n1 = x1.shape[1]
    x0_mean = np.mean(x0,axis=1)	
    x1_mean = np.mean(x1,axis=1)	
    
    x0_cov = np.cov(x0)	
    x1_cov = np.cov(x1)	
    #print("coverance matrix",x1_cov.shape)
    print("mean:",x0_mean.shape,x1_mean.shape)
    #print("x1_mean",x1_mean.shape)
    print("cov:",x0_cov.shape,x1_cov.shape)
    ratio = n0/(n0+n1)
    print("training data ratio",ratio)
    cov = ratio * x0_cov + (1 - ratio) * x1_cov 
    cov_inv = inv(cov)
    w = (x0_mean - x1_mean).T @ cov_inv 
    b = np.log(n0/n1) + (x1_mean.T@cov_inv@x1_mean - x0_mean.T@cov_inv@x0_mean) / 2	
    print('w',w.shape)
    return w,b 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__=="__main__":
    NP = "generative.npy" 
    train_x = sys.argv[1]
    train_y = sys.argv[2]
    test_x = sys.argv[3]
    output = sys.argv[4]
    training = sys.argv[5]

    if training == "train":
        print("Start training probabilistic generative model")
        x,y,x0,x1 = readtrainingdata(train_x,train_y)
        w,b = ProbabilisticGenerativeModel(x0,x1)
        print('x shape',x.shape)
        tmp = x@w + b
        train_predict = sigmoid(tmp)
        train_predict = [1 if v < 0.5 else 0 for v in train_predict]
        #print(train_predict)
        train_predict = (train_predict == y)
        train_predict = [1 if v == True else 0 for v in train_predict]
        #print(train_predict)
        acc = np.sum(train_predict) / len(y)
        print("training data accuracy",acc)

        para = np.concatenate((np.array([b]), w))
        print("Saving parameter to {}".format(NP))
        np.save(NP,para)
    
    print("Loading parameter to {}".format(NP))
    para = np.load(NP)
    b = para[0]
    w = para[1:]
    print("Start predicting by probabilistic generative model")
    predict(w,b,test_x,output)

