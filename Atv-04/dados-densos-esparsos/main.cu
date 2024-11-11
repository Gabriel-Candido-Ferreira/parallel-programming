#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

__global__ void matrixVectorMultiply(float *matrix, float *vector, float *result, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        float sum = 0.0;
        for (int col = 0; col < N; ++col) {
            sum += matrix[row * N + col] * vector[col];
        }
        result[row] = sum;
    }
}

void runTest(int size) {
    int M = size;
    int N = size;

    std::vector<float> h_matrix(M * N);
    std::vector<float> h_vector(N);
    std::vector<float> h_result(M, 0.0);

    srand(time(0));
    for (int i = 0; i < M * N; ++i) h_matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < N; ++i) h_vector[i] = static_cast<float>(rand()) / RAND_MAX;

    float *d_matrix, *d_vector, *d_result;
    cudaMalloc((void**)&d_matrix, M * N * sizeof(float));
    cudaMalloc((void**)&d_vector, N * sizeof(float));
    cudaMalloc((void**)&d_result, M * sizeof(float));

    cudaMemcpy(d_matrix, h_matrix.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, h_vector.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;

    auto start = std::chrono::high_resolution_clock::now();
    matrixVectorMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_vector, d_result, M, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    cudaMemcpy(h_result.data(), d_result, M * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Tamanho: " << size << " x " << size << "\n";
    std::cout << "Tempo CUDA: " << elapsed.count() << " segundos\n";

    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);
}

int main() {
    std::vector<int> sizes = {1000, 10000, 100000};
    for (int size : sizes) {
        runTest(size);
    }
    return 0;
}
