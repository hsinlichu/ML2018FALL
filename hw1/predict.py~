import sys,os
import numpy as np
import csv

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
    NPY = "./model/best.npy"
    
    select = [4,5,6,7,8,9,12]
    print("NPY name:",NPY)
    w = np.load(NPY)

    writeText = "id,value\n"
    testingData = readtestdata(testfile,select,HOURS,secondterm,U_bound,L_bound)
    result = predict(testingData,w,HOURS,select)
    for i in range(len(result)):
        writeText += "id_" + str(i) + "," + str(result[i][0]) + "\n"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        f.write(writeText)

if __name__ == '__main__':
    main(sys.argv)
