import numpy as np
import sys
import csv
'''
def draw(filename,x,y):
    plt.plot(x,y)
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.title("Loss")
    plt.grid(True)
    plt.savefig('%s.jpg'%filename)
    return None
'''
def OneHotEncoding(x):
    ret = []
    for r in x:
        tmp = []
        for i,c in enumerate(r):
            if i in [1]:
                encode = [0,0]
                encode[c - 1] = 1
                #tmp.extend(encode)
            elif i in [3]:
                encode = [0,0,0]
                encode[c - 1] = 1
                #tmp.extend(encode)
            elif i in [2]:
                encode = [0,0,0,0,0,0]
                encode[c - 1] = 1
                tmp.extend(encode)
            elif i in range(5,11):
                encode = [0 for c in range(11)]
                encode[c - 1] = 1
                tmp.extend(encode)
            else:
                tmp.append(i)
        ret.append(tmp)
    return ret  


def readdata(ftrain_x,ftrain_y,OneHot):
    train_x = None
    train_y = None
    val_x = None
    val_y = None
    whole_train_x = None
    whole_train_y = None
    '''
    with open("./data/ntrain_x.csv","r") as fin:           # Read training data
        train_x = list(csv.reader(fin))
        train_x = [[int(n) for n in r] for r in train_x]
        print("origin train_x data:",len(train_x))
    with open( "./data/ntrain_y.csv","r") as fin:
        train_y = list(csv.reader(fin))
        train_y = [[int(n[0])] for n in train_y]
        #print("origin train_y data:",len(train_y))
    with open("./data/nval_x.csv","r") as fin:             # Read validation data
        val_x = list(csv.reader(fin))
        val_x = [[int(n) for n in r] for r in val_x]
        print("origin val_x data:",len(val_x))
    with open( "./data/nval_y.csv","r") as fin:
        val_y = list(csv.reader(fin))
        val_y = [[int(n[0])] for n in val_y]
        #print("origin val_y data:",len(val_y))
    '''
    with open(ftrain_x,"r") as fin:            # Read whole training data 
        whole_train_x = list(csv.reader(fin))[1:]
        whole_train_x= [[int(n) for n in r] for r in whole_train_x]
        print("origin x data:",len(whole_train_x))
    with open(ftrain_y,"r") as fin:
        whole_train_y = list(csv.reader(fin))[1:]
        whole_train_y = [[int(n[0])] for n in whole_train_y]
        print("origin y data:",len(whole_train_y))

    if OneHot:
        '''
        train_x = OneHotEncoding(train_x)
        val_x = OneHotEncoding(val_x)
        '''
        whole_train_x = OneHotEncoding(whole_train_x)
    return train_x,train_y,val_x,val_y,whole_train_x,whole_train_y

def cross_entropy(predictions, targets, epsilon=1e-12):
    N = predictions.shape[0]
    loss = -np.sum(targets * np.log(predictions + epsilon) + (1 - targets) * np.log(1 - predictions + epsilon))/N
    return loss

def sigmoid(x):
    #print(x)
    return 1.0 / (1.0 + np.exp(-x))

def predict(w,b,x):
    #print("w.T",w.T)
    #print("b",b)
    #print("x",x)
    tmp = x@w+b 
    #print("(wx+b).T",tmp.T)
    return sigmoid(tmp)  

def accuracy(pd,y):
    correct = 0
    for a,b in zip(pd,y):
        if a == b:
            correct += 1
    return correct/len(y)

def mini_batch(x,y,batchsize):
    for i in range(0,len(x),batchsize):
        yield(x[i:i+batchsize][:],y[i:i+batchsize])

def logistic_regression(x,y,epoch,lr,batchsize):
    w = np.ones(len(x[0])).reshape(-1,1)
    b = 1
    ada_w = np.zeros(len(x[0])).reshape(-1,1)
    ada_b = 0
    epsilon = 1e-12
    print("w",w.shape)
    print("x",x.shape)
    xaxis = np.arange(1,epoch+1)
    history = []
    for i in range(epoch):
        for xb,yb in mini_batch(x,y,batchsize):
            N = len(xb) 
            pd = predict(w,b,xb)  
            #print("pd",pd.T)
            error = yb - pd 
            #print("error",error.T)
            #print("xb.T",xb.T)
            w_grad = -np.dot(xb.T,error) / N
            b_grad = -np.sum(error,axis=0) / N
            #print("w_grad",w_grad.T)
            #print("b_grad",b_grad)
            ada_w += w_grad**2
            ada_b += b_grad**2
            w -= (lr * w_grad / np.sqrt(ada_w + epsilon))
            b -= (lr * b_grad / np.sqrt(ada_b + epsilon))
        pd = predict(w,b,x)  
        loss = cross_entropy(pd,y)
        history.append(loss)
        print('\repoch: {}, Loss: {}'.format(i+1,loss), end='' ,flush=True)
    train_pd = predict(w,b,x)
    train_loss = cross_entropy(train_pd,y)
    train_pd = [1 if v >=0.5 else 0 for v in train_pd] 
    train_acc = accuracy(train_pd,y)
    print("\n\nTraining Loss: {} , Training accuracy: {}".format(train_loss,train_acc))
    '''
    draw("Logistic_regression_drop",xaxis,history)
    '''
    return w,b

if __name__ == "__main__":
    epoch = 1000
    lr = 0.3 
    batchsize = 100
    NP = "logistic_model.npy"
    OneHot = True
    train_x = sys.argv[1]
    train_y = sys.argv[2]
    test_x = sys.argv[3]
    output = sys.argv[4]
    training = sys.argv[5]
    
    if training == "train":
        train_x,train_y,val_x,val_y,whole_train_x,whole_train_y = readdata(train_x,train_y,OneHot)
        '''
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        val_x = np.array(val_x)
        val_y = np.array(val_y)
        '''
        whole_train_x = np.array(whole_train_x)
        whole_train_y = np.array(whole_train_y)

        '''
        w,b = logistic_regression(train_x,train_y,epoch,lr,batchsize)
        val_pd = predict(w,b,val_x)
        val_loss = cross_entropy(val_pd,val_y)
        val_pd = [1 if v >=0.5 else 0 for v in val_pd] 
        val_acc = accuracy(val_pd,val_y)
        print("Validation Loss: {} , Validation accuracy: {}".format(val_loss,val_acc))
        '''
        print("Start training on logistic regression")
        print("Epoch: {} | Learning rate: {} | batch size: {} | One hot encoding: {}| para: {}".format(epoch,lr,batchsize,OneHot,NP))
        w,b = logistic_regression(whole_train_x,whole_train_y,epoch,lr,batchsize)
        para = np.concatenate((np.array([b]), w))
        print("Saving parameter to {}".format(NP))
        np.save(NP,para)


    print("Loading parameter to {}".format(NP))
    para = np.load(NP)
    b = para[0][0]
    w = para[1:]
    test = None
    with open(test_x,"r") as fin:
        test = list(csv.reader(fin))[1:]
        test = [[int(n) for n in r] for r in test]
    if OneHot:
        test = OneHotEncoding(test)
    pd = predict(w,b,test)
    pd = pd > 0.5
    #print(predict)
    out = [['id','value']]
    for index,v in enumerate(pd):
        if v:
            out.append(['id_'+str(index),1])
        else:
            out.append(['id_'+str(index),0])
    print("Start predicting by logistic regression")
    with open(output,"w") as fout:
        writer = csv.writer(fout)
        writer.writerows(out)   
