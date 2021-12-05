import numpy as np
from numpy.linalg import norm
import os
import time
import random
import string


def get_similarity_matrix(adj_matrix , threshold=0.0002):
    threshold = threshold

    sys_random = random.SystemRandom()
    unique_time =  round(time.time() * 1000)
    random_string = ''.join(sys_random.choice(string.ascii_uppercase + string.digits + string.ascii_lowercase) for _ in range(5))
    path = './temp' + str(unique_time) + random_string

    # Transform to adjlist for deepwalk.
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[i])):
            if adj_matrix[i][j] == 1:
                adj_matrix[i][j] = int(j + 1)

    adj_matrix_copy = adj_matrix.tolist()
    adj_matrix_copy = [[int(value) for value in row if value != 0] for row in adj_matrix_copy]

    # First index represents the node, second index represents the neighbor
    adj_matrix_copy = [ [index + 1] + row for index, row in enumerate(adj_matrix_copy)]

    file_save_path = path + '.adjlist'
    adj_list = open(file_save_path, 'w')
    for row in adj_matrix_copy:
        adj_list.write(str(row)[1:-1].replace(",", "") + '\n')
    adj_list.close()

    output_path = "output" + str(unique_time) + random_string + ".embeddings"

    print("Running DeepWalk...")
    os.system("python3 deepwalk --input " + file_save_path + " --output " + output_path)

    embeddings = {}
    embeddings_file = open(output_path,'r')
    embeddings_file = embeddings_file.read().splitlines()

    print("Calculating similarity matrix...")
    for line_number, line in enumerate(embeddings_file):
        if(line_number == 0): continue
        line = line.split(" ")
        embeddings[int(line[0])] = np.array(line[1:]).astype(float)
        
    length_of_matrix = len(embeddings.keys())
    similarity_matrix = np.zeros((length_of_matrix, length_of_matrix)).astype(float)

    # Calculating norms.
    for key, value in embeddings.items():
        for i in range(1,length_of_matrix + 1):
            if(i == key): continue
            similarity_matrix[key-1][i-1] = norm(embeddings[key] - embeddings[i])

    # Normalizing norms.
    for i in range(length_of_matrix):
        sum_of_row = np.sum(similarity_matrix[i])
        similarity_matrix[i] = np.true_divide(similarity_matrix[i], sum_of_row)

    # min, max = np.amin(similarity_matrix), np.amax(similarity_matrix)
    # print("Min: " + str(min) + " Max: " + str(max))
    similarity_matrix = ((similarity_matrix <= threshold) & (similarity_matrix > 0)).astype(float)
    # np.savetxt("similarity_matrix", similarity_matrix, fmt='%d')
    return similarity_matrix

# Example usage.
# similarity_matrix = get_similarity_matrix('./example_graphs/cora')