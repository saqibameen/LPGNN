import torch
from torch_geometric.transforms import ToSparseTensor
from torch_sparse import SparseTensor
import numpy as np
from deepwalkprocessor import get_similarity_matrix 
import networkx as nx
from node2vec import Node2Vec


grarep_similarity_matrix = None
deep_walk_similarity_matrix = None
node2vec_similarity_matrix = None

def compute_similarity_matrix(similarity_method, adjacency_matrix):
    if(similarity_method == 'grarep'):
        return get_grarep_similarity_matrix(adjacency_matrix)
    elif(similarity_method == 'deepwalk'):
        # return get_deep_walk_similairy_matrix(adjacency_matrix.to_dense().cpu().numpy())
        return get_deep_walk_similairy_matrix(adjacency_matrix.to_dense().numpy())
    elif(similarity_method == 'node2vec'):
        return get_node2vec_similarity_matrix(adjacency_matrix)


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


def get_node2vec_similarity_matrix(adj_matrix, threshold = 0.999):

    global node2vec_similarity_matrix 
    if node2vec_similarity_matrix is not None:
        return node2vec_similarity_matrix
    else: 
        adj_matrix_dense = adj_matrix.to_dense()

        # Convert tensor to a numpy array
        adj_matrix_np = adj_matrix_dense.cpu().detach().numpy()

        # Get the graph in networkx format so that we can input it into the node2vec model
        G = nx.from_numpy_matrix(adj_matrix_np)

        # We can experiment with setting parameters here: https://github.com/eliorc/node2vec/blob/master/node2vec/node2vec.py
        node2vec = Node2Vec(G, dimensions=64, walk_length=5, p=0.5, q=3, num_walks=10, workers=1)

        # Get the model from the node2vec representation
        model = node2vec.fit()

        # A=model.wv.syn0 contains the list of vectors for each node in our graph
        # Therefore AA^T contains all the dot products between pairs of words

        # Normalize the node vectors so that each row has norm of 1 
        model.vectors = model.wv.syn0/np.linalg.norm(model.wv.syn0, axis=1)[:, None]
        similarity_matrix = np.dot(model.vectors, model.vectors.T)

        print(np.max(similarity_matrix, axis=1))
        print(np.min(similarity_matrix, axis=1))

        # Find which values are less than the threshold and cast it as a float 
        similarity_matrix = ((similarity_matrix >= threshold) & (similarity_matrix > 0)).astype(float)
        # Convert it to a form in torch 
        similarity_matrix_torch = torch.from_numpy(similarity_matrix)
        sparse_sim_matrix = similarity_matrix_torch.to_sparse()
        dense_sim_matrix = sparse_sim_matrix.to_dense()
        
        node2vec_similarity_matrix = SparseTensor.from_dense(dense_sim_matrix)
        return node2vec_similarity_matrix