import torch
from torch_geometric.transforms import ToSparseTensor
from torch_sparse import SparseTensor

grarep_similarity_matrix = None


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


        

    