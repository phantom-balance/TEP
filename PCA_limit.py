import pandas as pd
import torch
import os

class PCA():
    def __init__(self, Type, var_explained=None, confidence=None):
        self.Type = Type
        self.var_explained = var_explained
        self.confidence = confidence
        noc = pd.read_csv('TEP-profbraatz-dataset/d00.csv')
        self.mean = noc.mean()
        self.std = noc.std()
        noc_norm = noc.copy()
        for i in noc_norm:
            noc_norm[i] = (noc_norm[i]-self.mean[i])/(self.std[i])
        noc_norm = torch.tensor(noc_norm.values)
        noc_norm = torch.transpose(noc_norm, 0, 1)
        covariance_matrix = torch.cov(noc_norm)
        eigen_values, eigen_vectors = torch.linalg.eig(covariance_matrix)
        _, indices = torch.sort(eigen_values.abs(), dim=-1, descending=True)
        eigen_values_sorted = eigen_values.gather(dim=0, index=indices)
        eigen_vectors_sorted = eigen_vectors.gather(dim=0, index=indices.unsqueeze(-1).expand(52,52))
        total_var = eigen_values_sorted.sum()
        pca_var, num_components = 0, 0
        for i in range(52):
            v = eigen_values_sorted[i]/total_var
            v = v.abs()
            if pca_var < self.var_explained:
                pca_var += v
                num_components += 1
        truncated_eigen_vectors = eigen_vectors_sorted[:, :num_components]
        truncated_eigen_values = eigen_values_sorted[:num_components]
        noc_norm = torch.transpose(noc_norm, 0, 1)
        truncated_eigen_vectors = truncated_eigen_vectors.type(torch.double)
        PCA_mat = torch.mm(noc_norm, truncated_eigen_vectors)
        T2_PCA = torch.pow(PCA_mat, 2)
        T2_PCA = torch.div(T2_PCA, truncated_eigen_values)
        T2 = torch.sum(T2_PCA, dim=1).type(torch.double)
        self.UCL_T2 = torch.quantile(T2, self.confidence)
        Data_approximation = torch.mm(PCA_mat, torch.transpose(truncated_eigen_vectors,0,1))
        residual_matrix = noc_norm - Data_approximation
        Q = torch.mm(residual_matrix, torch.transpose(residual_matrix, 0, 1))
        Q = torch.diagonal(Q)
        self.UCL_Q = torch.quantile(Q, self.confidence)
        self.truncated_eigen_vectors = truncated_eigen_vectors
        self.truncated_eigen_values = truncated_eigen_values

    def return_func(self):
        Data_List = []
        for idx, num in enumerate(self.Type):
            data_path1 = os.path.join('TEP-profbraatz-dataset/', ('d0' if num<10 else 'd')+str(num)+".csv")
            data1 = pd.read_csv(data_path1)
            data_path2 = os.path.join('TEP-profbraatz-dataset/', ('d0' if num<10 else 'd')+str(num)+"_te.csv")
            data2 = pd.read_csv(data_path2)
            data1_norm = data1.copy()
            for i in data1_norm:
                data1_norm[i] = (data1_norm[i]-self.mean[i])/(self.std[i])
            data1_norm = torch.tensor(data1_norm.values)
            PCA1 = torch.mm(data1_norm, self.truncated_eigen_vectors)
            T2_PCA1 = torch.pow(PCA1, 2)
            T2_PCA1 = torch.div(T2_PCA1, self.truncated_eigen_values)
            T2_data1 = torch.sum(T2_PCA1, dim=1).type(torch.double)

            data2_norm = data2.copy()
            for i in data2_norm:
                data2_norm[i] = (data2_norm[i]-self.mean[i])/(self.std[i])
            data2_norm = torch.tensor(data2_norm.values)
            PCA2 = torch.mm(data2_norm, self.truncated_eigen_vectors)
            T2_PCA2 = torch.pow(PCA2, 2)
            T2_PCA2 = torch.div(T2_PCA2, self.truncated_eigen_values)
            T2_data2 = torch.sum(T2_PCA2, dim=1).type(torch.double)
            Data_Tuple = (T2_data1, T2_data2)
            Data_List.append(Data_Tuple)

        return self.UCL_T2, self.UCL_Q, Data_List


crap = PCA(Type=[0,1,2], var_explained=0.55, confidence=0.99)
print(crap.return_func())