#include <stdio.h>
#include<iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
}

void printDeviceArray(int* device_array, int N, const char* label) {
    int* host_array = new int[N];
    cudaMemcpy(host_array, device_array, N * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << label << ": ";
    for (int i = 0; i < N; i++) {
        std::cout << host_array[i] << " ";
    }
    std::cout << std::endl;
    delete[] host_array;
}

__global__ void downsweep(int* result,  int N, int stride) {    
  int threadIndex = threadIdx.x + (blockDim.x * blockIdx.x);
  int jump = stride * 2;
  int leftIndex = threadIndex * jump + stride - 1;
  int rightIndex = threadIndex * jump + jump - 1;
  int splits = N / jump;
  if(threadIndex < splits) {
    int temp = result[rightIndex];
    result[rightIndex] += result[leftIndex];
    result[leftIndex] = temp;
  }
}

__global__ void upsweep(int* result,  int N, int stride) {
  int threadIndex = threadIdx.x + (blockDim.x * blockIdx.x);
  int jump = stride * 2;
  int leftIndex = threadIndex * jump + stride - 1;
  int rightIndex = threadIndex * jump + jump - 1;
  int splits = N / jump;
  if(threadIndex < splits) {
    result[rightIndex] += result[leftIndex];
  }
}


void exclusive_scan(int* input, int N, int* result) {

  for(int stride = 1; stride < N; stride *= 2) {
    int strided = 2 * stride;
    int num_blocks = (N / strided + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // int num_blocks = N / strided;
    upsweep<<<num_blocks, THREADS_PER_BLOCK>>>(result, N, stride);
    cudaDeviceSynchronize();
  }

  int zero = 0;
  cudaMemcpy(result + N - 1, &zero, sizeof(int), cudaMemcpyHostToDevice);

  for(int stride = N/2; stride > 0; stride /= 2) {
    int strided = 2 * stride;
    int num_blocks = (N / strided + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // int num_blocks = N / strided;
    downsweep<<<num_blocks, THREADS_PER_BLOCK>>>(result, N, stride);
    cudaDeviceSynchronize();
  }
}

double cudaScan(int* inarray, int* end, int* resultarray)
{
  int* device_result;
  int* device_input;
  int N = end - inarray;  

  int rounded_length = nextPow2(end - inarray);
  
  cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
  cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

  cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

  double startTime = CycleTimer::currentSeconds();
  exclusive_scan(device_input, rounded_length, device_result);

  cudaDeviceSynchronize();
  double endTime = CycleTimer::currentSeconds();
      
  cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

  double overallDuration = endTime - startTime;
  return overallDuration; 
}

double cudaScanThrust(int* inarray, int* end, int* resultarray) {
  int length = end - inarray;
  thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
  thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
  
  cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

  double startTime = CycleTimer::currentSeconds();

  thrust::exclusive_scan(d_input, d_input + length, d_output);

  cudaDeviceSynchronize();
  double endTime = CycleTimer::currentSeconds();
  
  cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

  thrust::device_free(d_input);
  thrust::device_free(d_output);

  double overallDuration = endTime - startTime;
  return overallDuration; 
}


__global__ void map_repeats(int* input, int N, int* output) {
  int threadIndex = threadIdx.x + (blockDim.x * blockIdx.x);
  if(threadIndex < N - 1) {
    if(input[threadIndex] == input[threadIndex + 1]) {
      output[threadIndex] = 1;
    } else {
      output[threadIndex] = 0;
    }
  }
  if(threadIndex == N - 1) {
      output[threadIndex] = 0;
  }
}


__global__ void get_repeats(int* scan, int* output, int* flags, int length) {
  int threadIndex = threadIdx.x + (blockDim.x * blockIdx.x);
  if(threadIndex < length - 1 && flags[threadIndex] == 1) {
    output[scan[threadIndex]] = threadIndex;
  }
}


int find_repeats(int* device_input, int length, int* device_output) {
  int* device_flags;
  int* device_scan;
  int rounded_length = nextPow2(length);

  cudaMalloc((void **)&device_flags, sizeof(int) * rounded_length);
  cudaMalloc((void **)&device_scan, sizeof(int) * rounded_length);

  printDeviceArray(device_input, length, "DEVICE INPUT");

  // cudaMemcpy(device_input, device, N * sizeof(int), cudaMemcpyHostToDevice);
  // cudaMemcpy(device_result, inarray, N * sizeof(int), cudaMemcpyHostToDevice);
  int blocks = (length + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  map_repeats<<<blocks, THREADS_PER_BLOCK>>>(device_input, length, device_flags);
  cudaDeviceSynchronize();

  printDeviceArray(device_flags, length, "DEVICE MAP");

  cudaMemcpy(device_scan, device_flags, length * sizeof(int), cudaMemcpyHostToDevice);

  exclusive_scan(device_flags, rounded_length, device_scan);
  cudaDeviceSynchronize();

  printDeviceArray(device_scan, length, "DEVICE SCANNN");

  int total_repeats;
  cudaMemcpy(&total_repeats,  device_scan + length - 1, sizeof(int), cudaMemcpyDeviceToHost);

  get_repeats<<<blocks, THREADS_PER_BLOCK>>>(device_scan, device_output, device_flags, length);
  cudaDeviceSynchronize();

  cudaFree(device_flags);
  cudaFree(device_scan);
  
  return total_repeats;
}

double cudaFindRepeats(int *input, int length, int *output, int *output_length) {
  int *device_input;
  int *device_output;
  int rounded_length = nextPow2(length);
  
  cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
  cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
  cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  double startTime = CycleTimer::currentSeconds();
  
  int result = find_repeats(device_input, length, device_output);

  cudaDeviceSynchronize();
  double endTime = CycleTimer::currentSeconds();

  // set output count and results array
  *output_length = result;
  cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(device_input);
  cudaFree(device_output);

  float duration = endTime - startTime; 
  return duration;
}



void printCudaInfo()
{
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  printf("---------------------------------------------------------\n");
  printf("Found %d CUDA devices\n", deviceCount);

  for (int i=0; i<deviceCount; i++)
  {
      cudaDeviceProp deviceProps;
      cudaGetDeviceProperties(&deviceProps, i);
      printf("Device %d: %s\n", i, deviceProps.name);
      printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
      printf("   Global mem: %.0f MB\n",
              static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
      printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
  }
  printf("---------------------------------------------------------\n"); 
}
