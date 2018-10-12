import sys,os
import numpy as np
from numpy.linalg import inv
import csv
import matplotlib
matplotlib.use('agg')
import pylab as plt

def shuffle(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]

def closeForm(traingData, predict):
    close_weights = np.matmul(np.matmul(inv(np.matmul(traingData.T,traingData)),traingData.T),predict)
    return close_weights

def scaleData(traingData):
    v_min = np.min(traingData, axis=0)
    v_max = np.max(traingData, axis=0)
    return (traingData - v_min) / (v_max - v_min)

def readdata():
    data = [[] for i in range(18)]
    with open('./train.csv','r',encoding='big5') as text:
        row = csv.reader(text)
        ndata = list(row)
        for r in range(1,len(ndata)):
            for c in range(3,27):
                num = ndata[r][c]
                if num == 'NR':
                    num = 0.0
                data[(r-1) % 18].append(float(num))
        data = np.array(data)
        #data = rescaledata(data)
        #data = normalize(data)
        #print('read data array',data.shape)
    return data

def readtestdata(testfile,select,HOURS,secondterm,U_bound,L_bound):
    test = []
    with open(testfile,"r") as text:
        row = csv.reader(text,delimiter = ',')
        row = list(row)
        row = np.delete(row,0,1)
        row = np.delete(row,0,1)

        for count,r in enumerate(row):
            if count % 18 == 0:
                test.append([])
            if (count % 18) in select:
                for i in range(9 - HOURS,9):
                    tmp = r[i]
                    if r[i] == "NR":
                        tmp = 0
                    tmp = float(tmp)
                    if count%18 == 9:
                        #print(tmp)
                        if tmp < L_bound:
                            pass
                            #tmp = L_bound
                        elif tmp > U_bound:
                            tmp = U_bound
                    test[count // 18].append(tmp)
        
    test = np.array(test).reshape((260,-1))
    for x,r in enumerate(test):
        for y,c in enumerate(r):
            if test[x][y] <= 0:
                if y == 0:
                    test[x][y] = test[x][y+1]
                elif y == test.shape[1] - 1:
                    test[x][y] = test[x][y-1]
                else:
                    test[x][y] = (test[x][y+1] + test[x][y-1])/2
    if secondterm:
        test = np.concatenate((test,test**2),axis=1) #add second order
    #test = np.concatenate((np.ones((test.shape[0],1)),test), axis=1) # add bias
    #test = np.insert(test,0,1,axis=1)
    #print('test',test.shape)
    return test

def parsedata(odata, HOURS, select,secondterm,U_bound,L_bound):
    #print("parse odata",odata.shape)
    feature = []
    label = []
    for m in range(12): #month
        for group in range(20*24-HOURS):
            tmp = []
            start = m*20*24 + group
            choose = True
            #tmp = np.array(odata[:,start:start + HOURS])
            for s in select:
                tmp.append(odata[s][start:start + HOURS])
                if s == 9:
                    for d in odata[s][start:start + HOURS + 1]:
                        if d > U_bound or d < L_bound:
                            choose = False
                            break
                else:
                    for d in odata[s][start:start + HOURS + 1]:
                        if d == 0:
                            choose = False
                            break
            if choose:
                feature.append(tmp)
                label.append(odata[9][start + HOURS])
    feature = np.array(feature)
    label = np.reshape(np.array(label),(-1,1))
    #print('origin feature',feature.shape)
    #print('label',label.shape)
    
    feature = np.array([case.flatten() for case in feature])
    if secondterm:
        feature = np.concatenate((feature,feature**2),axis=1) #add second order
    #feature = np.concatenate((np.ones((feature.shape[0],1)),feature),axis=1) #add bias
    #feature = np.insert(feature,0,1,axis=1) # add bias
    print('feature shape',feature.shape)
    feature,label = shuffle(feature,label)
    return feature,label

def calculateError(traingData, predict, w, b):
    return (np.dot(traingData, w) + b) - predict

def calculateLoss(traingData, predict, w, b, lamda):
    return np.sqrt(np.mean(calculateError(traingData, predict, w, b) ** 2) + lamda * np.sum(w ** 2))

def adagrad(traingData, predict, LEARNING_RATE, ITERATION, HOURS, LAMDA):
    print("Start running adagrad")
    b = 0.0
    w = np.ones((len(traingData[0]), 1))
    B_lr = 0.0  #Adagrad
    W_lr = np.zeros((len(traingData[0]), 1)) #Adagrad
    N = len(traingData)
    xaxis = np.arange(1,ITERATION + 1)
    history = []
    for index in range(ITERATION):
        error = calculateError(traingData, predict, w, b)
        B_grad = np.sum(error) * 1.0 / N
        #W_grad = np.dot(traingData.T, error) / N # renew each weights
        W_grad = np.dot(traingData.T, error) / N + 2*LAMDA*w # renew each weights
        
        B_lr += B_grad ** 2
        W_lr += W_grad ** 2

        b -= LEARNING_RATE / np.sqrt(B_lr) * B_grad
        #w = w * (1 - (LEARNING_RATE / np.sqrt(W_lr)) * LAMDA) - LEARNING_RATE / np.sqrt(W_lr) * W_grad
        w -= (LEARNING_RATE / np.sqrt(W_lr)) * W_grad 
        current_loss = calculateLoss(traingData, predict, w, b, LAMDA)
        history.append(current_loss)
        print('\rIteration: {}, Loss: {}'.format(str(index+1), current_loss), end='' ,flush=True)
    print()
    plt.plot(xaxis[3:],history[3:])
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.title("Loss")
    plt.grid(True)
    plt.savefig('Loss.jpg')
    return np.concatenate((np.array([[b]]), w))


def predict(testingData, w, HOURS, select):
    result = []
    for i in range(len(testingData)):
        result.append(np.dot(testingData[i], w[1:]) + w[0][0])
    return result

def main(argv):
    testfile = argv[1]
    output = argv[2]
    ITERATION = 50000
    LEARNING_RATE = 0.5
    DATA_CATEGORIES = 18
    HOURS = 6
    LAMDA = 0.0
    SPLIT = 0.9
    secondterm = True
    U_bound = 120
    L_bound = 2
    NPY = "./model/test.npy"
    
    select = [4,5,6,8,9,12]
    #select = [9]
    # select = np.array(range(18))

    print("Period %d | lr %f | Hour %d | Lamda %f | SPLIT %f" %(ITERATION,LEARNING_RATE,HOURS,LAMDA,SPLIT))
    print("Select feature:",select)
    print("Second Term:",secondterm)
    print("Upper bound %d | Lower bound %d"%(U_bound,L_bound))
    print("NPY name:",NPY)
    
    getData = readdata()
    feature,label = parsedata(getData, HOURS, select,secondterm,U_bound,L_bound)
    
    cut = int(SPLIT*len(feature))
    train_x = feature[:cut,:]
    train_y = label[:cut,0].reshape((-1,1))
    val_x = feature[cut:,:]
    val_y = label[cut:,0].reshape((-1,1))                       
    print('train_x',train_x.shape,'val_x',val_x.shape)
    #print('train_y',train_y.shape)
    #print('val_x',val_x.shape)
    #print('val_y',val_y.shape)
    
    close_weights = closeForm(train_x, train_y)
    close_loss = calculateLoss(train_x, train_y, close_weights, 0.0, LAMDA)
    print("Close form {}".format(close_loss))

    w = adagrad(train_x, train_y, LEARNING_RATE, ITERATION, HOURS, LAMDA)
    validation_Loss = calculateLoss(val_x, val_y, w[1:], w[0][0], LAMDA)
    print("Validation Loss {}".format(validation_Loss))

    #print("Retrain on whole data")
    train_x = feature
    train_y = label
    w = adagrad(train_x, train_y, LEARNING_RATE, ITERATION, HOURS, LAMDA)
    np.save(NPY,w)

    w = np.load(NPY)

    writeText = "id,value\n"
    testingData = readtestdata(testfile,select,HOURS,secondterm,U_bound,L_bound)
    #testingData = scaleData(testingData)
    result = predict(testingData,w,HOURS,select)
    for i in range(len(result)):
        writeText += "id_" + str(i) + "," + str(result[i][0]) + "\n"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        f.write(writeText)


if __name__ == '__main__':
    main(sys.argv)
