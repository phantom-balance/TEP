from getdata import get_data
import torch
from torch.utils.data import Dataset

class TEP(Dataset):
    def __init__(self, num, sequence_length, is_train=None):
        self.Lis = get_data(num, is_train=is_train)
        self.sequence_length = sequence_length
        k = 0
        for i in range(len(self.Lis)):
            if i % 2 != 0:
                list_class = self.Lis[i]
                list_instance = list_class[0]
                k = k + len(list_instance) - self.sequence_length + 1
            else:
                pass
        self.length = k

    def __len__(self):
        length = self.length
        return length

    def __getitem__(self, index):
        classification_list = []
        new_df_list = []
        for i in range(len(self.Lis)):
            if i % 2 != 0:
                list_class = self.Lis[i]
                list_instance = list_class[0]
                temp_list_instance = list_instance.copy()
                for j in range(self.sequence_length-1):
                    temp_list_instance.pop(0)
                classification_list = classification_list + temp_list_instance

            else:
                df_class = self.Lis[i]
                df_instance = df_class[0]
                df_instance = df_instance.values.tolist()
                temp_df_list = df_instance.copy()
                for i in range(len(df_instance)-self.sequence_length+1):
                    for j in range(self.sequence_length):
                        new_df_list.append(temp_df_list[j])
                    temp_df_list.pop(0)

        classification_tensor = torch.tensor(classification_list)
        df_tensor = torch.tensor(new_df_list)
        df_tensor = df_tensor.reshape(self.length, self.sequence_length, 52)

        seq_data = df_tensor[index]
        label = classification_tensor[index]

        return (seq_data, label)