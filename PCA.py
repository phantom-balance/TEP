import pandas as pd
import torch
from scipy import stats


noc = pd.read_csv('TEP-profbraatz-dataset/d00.csv')
noc_te = pd.read_csv('TEP-profbraatz-dataset/d00_te.csv')

# print("noc:",noc)
# normalize using mean and std from noc training

mean = noc.mean()
std = noc.std()

noc_norm = noc.copy()
for i in noc_norm:
    noc_norm[i] = (noc_norm[i]-mean[i])/(std[i])
##############################################################################
noc_te_norm = noc_te.copy()
for i in noc_te_norm:
    noc_te_norm[i] = (noc_te_norm[i]-mean[i])/(std[i])
noc_te_norm = torch.tensor(noc_te_norm.values)
##############################################################################

# print("noc_norm:",noc_norm) #precise
# calculate covariance matrix
noc_norm = torch.tensor(noc_norm.values)
noc_norm = torch.transpose(noc_norm, 0, 1)
covariance_matrix = torch.cov(noc_norm)
# print("covariance_matrix:",covariance_matrix) #error 0.000005

# eigenvalue decomposition of the covariance matrix
eigen_values, eigen_vectors = torch.linalg.eig(covariance_matrix)

# descending order arrangement of the eigenvalues
_, indices = torch.sort(eigen_values.abs(), dim=-1, descending=True)
# print("indices:",indices)
eigen_values_sorted = eigen_values.gather(dim=0, index=indices)
eigen_vectors_sorted = eigen_vectors.gather(dim=0, index=indices.unsqueeze(-1).expand(52,52))

# print("eigen_values:",eigen_values) # some error
# print("eigen_vectors:",eigen_vectors) # some error
# print("eigen_values_sorted:",eigen_values_sorted)
# print("eigen_vectors_sorted:",eigen_vectors_sorted)

# determination of the number of eigenvectors for PCA
total_var = eigen_values_sorted.sum()

pca_var, num_components, var_explained = 0, 0, 0.55
for i in range(52):
    v = eigen_values_sorted[i]/total_var
    v = v.abs()
    if pca_var < var_explained:
        pca_var += v
        num_components += 1
# print("PC variability explained, num of PC"pca_var, num_components) # some error

truncated_eigen_vectors = eigen_vectors_sorted[:, :num_components]
truncated_eigen_values = eigen_values_sorted[:num_components]
# print("truncated_eigen_vectors:", truncated_eigen_vectors)
# print("truncated_eigen_values:", truncated_eigen_values)

# Projecting our data into the new truncated_eigen_vectors
noc_norm = torch.transpose(noc_norm, 0, 1)
truncated_eigen_vectors = truncated_eigen_vectors.type(torch.double)
PCA_mat = torch.mm(noc_norm, truncated_eigen_vectors)
###########################################################################
PCA_mat2 = torch.mm(noc_te_norm, truncated_eigen_vectors)
T2_PCA2 = torch.double
T2_PCA2 = torch.pow(PCA_mat2, 2)
T2_PCA2 = torch.div(T2_PCA2, truncated_eigen_values)
# print("##",PCA_mat)
T2_2 = torch.sum(T2_PCA2, dim=1).type(torch.double)
UCL_T2_2 = torch.quantile(T2_2, 0.99)
print("UCL_T2_2", UCL_T2_2)
###########################################################################
# print("truncated_eigen_vectors:", truncated_eigen_vectors)#[:,10:]) # large error(sign flip-->2nd last/last)
# print("PCA projection:", PCA_mat) # larger error(even sign flip-->2nd last/last)


# T2 calculation of the noc/pca data
# print("PCA_mat.shape",PCA_mat.shape)
# mean_PCA = torch.mean(PCA_mat, dim=0)
# cov_PCA = torch.cov(torch.transpose(PCA_mat, 0, 1))
# cov_PCA_inv = torch.inverse(cov_PCA)
# std_PCA = torch.std(PCA_mat)
# print("mean_PCA",mean_PCA)
# print("cov_PCA",cov_PCA.shape)
# print("std_PCA", std_PCA)

T2_PCA = torch.double
T2_PCA = torch.pow(PCA_mat, 2)
# print("T2_PCA",T2_PCA)

T2_PCA = torch.div(T2_PCA, truncated_eigen_values)
T2 = torch.sum(T2_PCA, dim=1).type(torch.double)
# print("T2 shape",T2.shape)
UCL_T2 = torch.quantile(T2, 0.99)
print("UCL_T2", UCL_T2)

# Q calculation of the noc/pca data
Data_approximation = torch.mm(PCA_mat, torch.transpose(truncated_eigen_vectors,0,1))
# print("Data_approximation",Data_approximation.shape)

residual_matrix = noc_norm - Data_approximation
# print("residual_matrix", residual_matrix)

Q = torch.mm(residual_matrix, torch.transpose(residual_matrix, 0, 1))
Q = torch.diagonal(Q)
UCL_Q = torch.quantile(Q, 0.99)
print("UCL_Q", UCL_Q)
# control chart plot ---> do it in colab after passing data from here

