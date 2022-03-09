import pandas as pd
import torch


noc = pd.read_csv('TEP-profbraatz-dataset/d00.csv')
noc_te = pd.read_csv('TEP-profbraatz-dataset/d00_te.csv')

# print("noc:",noc)
# normalize using mean and std from noc training

mean = noc.mean()
std = noc.std()

noc_norm = noc.copy()
for i in noc_norm:
    noc_norm[i] = (noc_norm[i]-mean[i])/(std[i])

# print("noc_norm:",noc_norm)
# calculate covariance matrix
noc_norm = torch.tensor(noc_norm.values)
noc_norm = torch.transpose(noc_norm, 0, 1)
covariance_matrix = torch.cov(noc_norm)
# print("covariance_matrix:",covariance_matrix)

# eigenvalue decomposition of the covariance matrix
eigen_values, eigen_vectors = torch.linalg.eig(covariance_matrix)

# descending order arrangement of the eigenvalues
_, indices = torch.sort(eigen_values.abs(), dim=-1, descending=True)
# print("indices:",indices)
eigen_values_sorted = eigen_values.gather(dim=0, index=indices)
eigen_vectors_sorted = eigen_vectors.gather(dim=0, index=indices.unsqueeze(-1).expand(52,52))

# print("eigen_values:",eigen_values)
# print("eigen_vectors:",eigen_vectors)
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
# print(pca_var, num_components)

truncated_eigen_vectors = eigen_vectors_sorted[:, :num_components]
# print("truncated_eigen_vectors:", truncated_eigen_vectors)

# Projecting our data into the new truncated_eigen_vectors
noc_norm = torch.transpose(noc_norm, 0, 1)
truncated_eigen_vectors = truncated_eigen_vectors.type(torch.double)
PCA_mat = torch.mm(noc_norm, truncated_eigen_vectors)
# print("truncated_eigen_vectors:", truncated_eigen_vectors)
print("PCA projection:", PCA_mat)


# T2 calculation of the noc/pca data
# Q calculation of the noc/pca data
# control chart plot
# failure testing with the T2 and Q from the noc-pca data

