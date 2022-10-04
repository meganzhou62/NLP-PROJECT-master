import csv
import numpy as np
import matplotlib.pyplot as plt

data_path1 = 'results/resultfoffnnh=32lr=0.01act=relu.csv'
data_path2 = 'results/resultfornn=32lr=0.01act=relu.csv'
def load(datapath):
    with open(datapath, 'r') as f:
        reader = csv.reader(f, delimiter=',', )
        # get header from first row
        # get all the rows as a list
        data = list(reader)
        # transform data into numpy array
        data = np.array(data).astype(float)
        if len(data)<10:
            data = np.transpose(data)

    return data[data[:,0].argsort()]
data1 = load(data_path1)
data2 = load(data_path2)
print(data1)
print(data2)
N = 0
P=0.0
for i in range(len(data1)):
    if data1[i][1]==data1[i][2]:
        if data2[i][1]==data2[i][2]:
            P+=1
        N+=1
print(P/float(N))

data_path3 = 'results/resultfoffnnh=64lr=0.01act=relu.csv'
data_path4 = 'results/resultfornn=64lr=0.01act=relu.csv'
data3 = load(data_path3)
data4 = load(data_path4)

N = 0
P=0.0
for i in range(len(data1)):
    if data3[i][1]==data3[i][2]:
        if data4[i][1]==data4[i][2]:
            P+=1
        N+=1
print(P/float(N))


data_path5 = 'results/resultfoffnnh=32lr=0.001act=relu.csv'
data_path6 = 'results/resultfornn=32lr=0.001act=relu.csv'
data5 = load(data_path5)
data6 = load(data_path6)

N = 0
P=0.0
for i in range(len(data1)):
    if data5[i][1]==data5[i][2]:
        if data6[i][1]==data6[i][2]:
            P+=1
        N+=1
print(P/float(N))