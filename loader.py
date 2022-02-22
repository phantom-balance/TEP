from getdata import get_data
import torch
from torch.utils.data import Dataset


class TEP(Dataset):
    def __init__(self, num, is_train=None):
        Lis = get_data(num, is_train=is_train)
        self.Lis = Lis

    def __len__(self):
        k = int(0)
        for i in range(len(self.Lis)):
            if i % 2 != 0:
                list_class = self.Lis[i]
                list_instance = list_class[0]
                k += len(list_instance)
        return k

    def __getitem__(self, index):
        classification_list = []
        df_list = []
        for i in range(len(self.Lis)):
            if i % 2 != 0:
                list_class = self.Lis[i]
                list_instance = list_class[0]
                classification_list = classification_list + list_instance
            else:
                df_class = self.Lis[i]
                df_instance = df_class[0]
                df_instance = df_instance.values.tolist()
                df_list = df_list + df_instance
        classification_tensor = torch.tensor(classification_list)
        df_tensor = torch.tensor(df_list)
        label = classification_tensor[index]
        data_instance = df_tensor[index]

        return (data_instance, label)