#include <iostream>
#include <cuda.h>
#include <vector>
#include <ctime>


#define NUM_BUCKETS 10


__global__ void histogramKernel(float *data, int *histogram, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements) {
        float value = data[idx];
        if (value >= 0.0f && value <= 10.0f) {
            int bin = min(static_cast<int>(value), NUM_BUCKETS - 1);
            atomicAdd(&histogram[bin], 1);
        }
    }
}

int main() {

    int numElements = 100000000;
    std::vector<float> hostData(numElements);
    srand(time(0));
    for (int i = 0; i < numElements; ++i) {
        hostData[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX/10.0));
    }


    int hostHistogram[NUM_BUCKETS] = {0};


    float *deviceData;
    int *deviceHistogram;
    cudaMalloc((void**)&deviceData, numElements * sizeof(float));
    cudaMalloc((void**)&deviceHistogram, NUM_BUCKETS * sizeof(int));


    cudaMemcpy(deviceData, hostData.data(), numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceHistogram, hostHistogram, NUM_BUCKETS * sizeof(int), cudaMemcpyHostToDevice);


    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    histogramKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceData, deviceHistogram, numElements);
    cudaDeviceSynchronize();


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);


    cudaMemcpy(hostHistogram, deviceHistogram, NUM_BUCKETS * sizeof(int), cudaMemcpyDeviceToHost);


    int totalSum = 0;
    for (int i = 0; i < NUM_BUCKETS; ++i) {
        totalSum += hostHistogram[i];
    }


    std::cout << "Histograma:\n";
    for (int i = 0; i < NUM_BUCKETS; ++i) {
        std::cout << "Intervalo " << i << ": " << hostHistogram[i] << "\n";
    }
    std::cout << "Soma total de todas as faixas: " << totalSum << "\n";

    std::cout << "Tempo de processamento (CUDA): " << milliseconds / 1000.0 << " segundos\n";



    cudaFree(deviceData);
    cudaFree(deviceHistogram);

    return 0;
}
