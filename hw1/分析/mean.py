import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def readdata():
    data = [[] for i in range(18)]
    with open('../train.csv','r',encoding='big5') as text:
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
                        if d > 120 or d < 2:
                            choose = False
                            break
            if choose:
                feature.append(tmp)
                if odata[9][start + HOURS] < 0:
                    print(odata[9][start + HOURS])
                label.append(odata[9][start + HOURS])
    feature = np.array(feature)
    label = np.reshape(np.array(label),(-1,1))
    feature = np.array([case.flatten() for case in feature])
    if secondterm:
        feature = np.concatenate((feature,feature**2),axis=1) #add second order
    return feature,label

HOURS = 9
LAMDA = 0.0
SPLIT = 0.9
secondterm = False
U_bound = 110
L_bound = 2

select = [9]
getData = readdata()
feature,label = parsedata(getData, HOURS, select,secondterm,U_bound,L_bound)
mean = feature.mean(axis=1)
print(mean.shape)

f = plt.figure(1)
x = np.arange(len(mean))
plt.plot(label,mean,'.')
plt.ylabel('9hr mean')
plt.xlabel('label')
plt.title("PM2.5")
f.savefig('pn2.5_mean.jpg')
