import pandas as pd
import torch
import numpy as np

noc = pd.read_csv('TEP-profbraatz-dataset/d00.csv')
mean = noc.mean()
std = noc.std()
noc_norm = noc.copy()
for i in noc_norm:
    noc_norm[i] = (noc_norm[i]-mean[i])/(std[i])
noc_norm = torch.tensor(noc_norm.values)
noc_norm = torch.transpose(noc_norm, 0, 1)
covariance_matrix = torch.cov(noc_norm)
eigen_values, eigen_vectors = torch.linalg.eig(covariance_matrix)
_, indices = torch.sort(eigen_values.abs(), dim=-1, descending=True)
eigen_values_sorted = eigen_values.gather(dim=0, index=indices)
eigen_vectors_sorted = eigen_vectors.gather(dim=0, index=indices.unsqueeze(-1).expand(52,52))
total_var = eigen_values_sorted.sum()
pca_var, num_components, var_explained = 0, 0, 0.56
for i in range(52):
    v = eigen_values_sorted[i]/total_var
    v = v.abs()
    if pca_var < var_explained:
        pca_var += v
        num_components += 1
print("#", num_components)
truncated_eigen_vectors = eigen_vectors_sorted[:, :num_components]
truncated_eigen_values = eigen_values_sorted[:num_components]
noc_norm = torch.transpose(noc_norm, 0, 1)
truncated_eigen_vectors = truncated_eigen_vectors.type(torch.double)
PCA_mat = torch.mm(noc_norm, truncated_eigen_vectors)
T2_PCA = torch.double
T2_PCA = torch.pow(PCA_mat, 2)
T2_PCA = torch.div(T2_PCA, truncated_eigen_values)
T2 = torch.sum(T2_PCA, dim=1).type(torch.double)
UCL_T2 = torch.quantile(T2, 0.99)
print("UCL_T2", UCL_T2)
FDR = 0
for i in range(500): #ranges according to file determine garnu parne xa
    if T2[i] > UCL_T2:
        # print(i)
        FDR+=1
print("FDR",FDR)
Data_approximation = torch.mm(PCA_mat, torch.transpose(truncated_eigen_vectors,0,1))
residual_matrix = noc_norm - Data_approximation
Q = torch.mm(residual_matrix, torch.transpose(residual_matrix, 0, 1))
Q = torch.diagonal(Q)
UCL_Q = torch.quantile(Q, 0.99)
print("UCL_Q", UCL_Q)

# For other data
noc_te = pd.read_csv('TEP-profbraatz-dataset/d00_te.csv')
noc_te_norm = noc_te.copy()
for i in noc_te_norm:
    noc_te_norm[i] = (noc_te_norm[i]-mean[i])/(std[i])
noc_te_norm = torch.tensor(noc_te_norm.values)
# noc_te_norm = torch.transpose(noc_te_norm, 0, 1)
new_PCA_mat = torch.mm(noc_te_norm, truncated_eigen_vectors)
T2_PCA = torch.double
T2_PCA = torch.pow(new_PCA_mat, 2)
T2_PCA = torch.div(T2_PCA, truncated_eigen_values)
T2 = torch.sum(T2_PCA, dim=1).type(torch.double)

FDR = 0
for i in range(960): #ranges according to file determine garnu parne xa
    if T2[i] > UCL_T2:
        FDR+=1
print("FDR",FDR)
UCL_T2 = torch.quantile(T2, 0.99)
print("UCL_T2", UCL_T2)
