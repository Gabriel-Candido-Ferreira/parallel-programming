import numpy as np
import time

def serial_matrix_vector_multiplication(matrix, vector):
    return np.dot(matrix, vector)

# Teste com tamanhos menores
sizes = [1000, 10000, 100000]

for size in sizes:
    print(f"Tamanho: {size} x {size}")
    matrix = np.random.rand(size, size).astype(np.float32)
    vector = np.random.rand(size).astype(np.float32)

    start_time = time.time()
    result = serial_matrix_vector_multiplication(matrix, vector)
    end_time = time.time()

    print(f"Tempo Serial: {end_time - start_time:.6f} segundos\n")
