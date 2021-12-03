import torch
from torch_geometric.transforms import ToSparseTensor
from torch_sparse import SparseTensor
import numpy as np
from deepwalkprocessor import get_similarity_matrix 


grarep_similarity_matrix = None
deep_walk_similarity_matrix = None

def compute_similarity_matrix(similarity_method, adjacency_matrix):
    if(similarity_method == 'grarep'):
        return get_grarep_similarity_matrix(adjacency_matrix)
    elif(similarity_method == 'deepwalk'):
        return get_deep_walk_similairy_matrix(adjacency_matrix.to_dense().cpu().numpy())


def get_grarep_similarity_matrix(adj_t, transition_steps=10, similarity_threshold=0.01):     
    global grarep_similarity_matrix 
    if grarep_similarity_matrix is not None:
        return grarep_similarity_matrix
    else:
        adj_tensor = adj_t.to_dense()
        adj_tensor_org = torch.clone(adj_tensor)
        final_adj_tensor = torch.clone(adj_tensor)
        product_tensor = torch.clone(adj_tensor)
        for step in range(2, transition_steps+1):
            product_tensor = product_tensor + torch.linalg.matrix_power(adj_tensor_org, step)
        product_tensor = product_tensor / (torch.sum(product_tensor, 1).unsqueeze(-1))
        product_tensor = torch.where(product_tensor>similarity_threshold,1,0)
        for row,row_tensor in enumerate(product_tensor):
            for col, col_tensor in enumerate(row_tensor):
                if product_tensor[row,col] > 0:
                    final_adj_tensor[row] = final_adj_tensor[row]+adj_tensor[col]

        final_adj_tensor = torch.where(final_adj_tensor>0,1.0,0.0)
        grarep_similarity_matrix = SparseTensor.from_dense(final_adj_tensor)
        #grarep_similarity_matrix = adj_t
        return grarep_similarity_matrix

def get_deep_walk_similairy_matrix(adj_matrix):
    global deep_walk_similarity_matrix 
    if deep_walk_similarity_matrix is not None:
        return deep_walk_similarity_matrix
    else:
        adj_matrix = torch.from_numpy(get_similarity_matrix(adj_matrix))
        sparse_adj_matrix = adj_matrix.to_sparse()
        dense_adj_matrix = sparse_adj_matrix.to_dense()
        deep_walk_similarity_matrix = SparseTensor.from_dense(dense_adj_matrix)
        return deep_walk_similarity_matrix


