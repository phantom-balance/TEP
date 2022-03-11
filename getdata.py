import os
import pandas as pd

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
                for i in range(0,480):
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

