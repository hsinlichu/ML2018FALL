import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("../test.csv",'r',encoding='big5') as fin:
    row = np.array(list(csv.reader(fin)))
    #row = np.delete(row,0,0)
    title = [row[i][1] for i in range(18)]
    print(title)
    for i in range(2):
        row = np.delete(row,0,1)
    data = [[] for i in range(18)]
    for cnt,r in enumerate(row):
        for c in r:
            tmp = c
            if tmp == "NR":
                tmp = 0
            data[cnt%18].append(float(tmp))
    data = np.array(data)
    print(data.shape)

    '''    
    x = []
    y = []
    for m in range(12):
        for i in range(479):
            x.append(data[9][480*m+i])
            y.append(data[9][480*m+i+1])
    '''
    x = np.arange(len(data[0]))
    for i in range(18):
        f = plt.figure(i)
        plt.title(title[i])
        plt.plot(x,data[i])
        f.savefig('./testing/%d.jpg'%i)
    '''
    f = plt.figure(1)
    plt.plot(x,y,'.')
    plt.ylabel('now data')
    plt.xlabel('previous data')
    plt.title("PM2.5")
    f.show()
    f.savefig('parse.jpg')
    '''
