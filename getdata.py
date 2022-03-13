import os
import pandas as pd
'''
Returns Data inside a list and the labels for every time step
of the Data in a list in that order.
example:
get_data([1,2], is_train=True) 
will return ([data1],[labels-for-data1],[data2],[labels-for-data2])
[data1]--->[d01.csv] 
[labels-for-data1]--->[1,1,......,1]{480 labels, labeled 1}
when is_trian=False:
[data1]--->[d01_te.csv]
[labels-for-data1]--->[1,1,......,1]{160 labels 0, 161 to 960 labeled 1}
'''


def get_data(list, is_train=None):
    if is_train==True:
        k = []
        for idx,num in enumerate(list):
            data_path = os.path.join('TEP-profbraatz-dataset/', ('d0' if num<10 else 'd')+str(num)+".csv")
            data = pd.read_csv(data_path)
            if data_path == 'TEP-profbraatz-dataset/d00.csv':
               m = []
               for i in range (500):
                   m.append(0)
               k.extend([[data],[m]])

            else:
                m = []
                for i in range(480):
                    m.append(num)
                k.extend([[data],[m]])

    else:
        k = []
        for idx,num in enumerate(list):
            data_path = os.path.join('TEP-profbraatz-dataset/', ('d0' if num<10 else 'd')+str(num)+"_te.csv")
            data = pd.read_csv(data_path)

            if data_path == "TEP-profbraatz-dataset/d00_te.csv":
                m = []
                for i in range(960):
                    m.append(0)
                k.extend([[data],[m]])

            else:
                m = []
                for i in range(160):
                    m.append(0)
                for i in range(160,960):
                    m.append(num)
                k.extend([[data],[m]])

    return k

