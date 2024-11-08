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

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result


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
  // // Ensure only the first thread of the first block sets the last element to 0
  int threadIndex = threadIdx.x + (blockDim.x * blockIdx.x);
  // if (threadIndex == 0) {
  //     result[N - 1] = 0;
  // }
  // __syncthreads();
  // for(int stride = N/2; stride > 0; stride /= 2) {
    int jump = stride * 2;
    int leftIndex = threadIndex * jump + stride - 1;
    int rightIndex = threadIndex * jump + jump - 1;
    int splits = N / jump;
    if(threadIndex < splits) {
      int temp = result[rightIndex];
      result[rightIndex] += result[leftIndex];
      result[leftIndex] = temp;
    }
    // __syncthreads();
  // }
}

__global__ void upsweep(int* result,  int N, int stride) {
  int threadIndex = threadIdx.x + (blockDim.x * blockIdx.x);
  // for(int stride = 1; stride < N; stride *= 2) {
    int jump = stride * 2;
    int leftIndex = threadIndex * jump + stride - 1;
    int rightIndex = threadIndex * jump + jump - 1;
    int splits = N / jump;
    if(threadIndex < splits) {
      result[rightIndex] += result[leftIndex];
    }
    // __syncthreads();
  // }
}

// __global__ void upsweep(int* result, int N, int stride) {
//   int threadId = threadIdx.x + (blockDim.x * blockIdx.x);
//   int strided = stride * 2;
//   int splits = N / strided;
//   if(threadId < splits) {
//     int jump = threadId * strided;
//     result[jump + strided - 1] += result[jump + stride - 1];
//   }
//   __syncthreads();
// }

// __global__ void downsweep(int* result, int N, int stride) {
//   int threadId = threadIdx.x + (blockDim.x * blockIdx.x);
//   int strided = stride * 2;
//   int splits = N / strided;
//   if(threadId < splits) {
//     int jump = threadId * strided;
//     int temp = result[jump + strided - 1];
//     result[jump + strided - 1] += result[jump + stride - 1];
//     result[jump + stride - 1] = temp;
//   }
//   __syncthreads();
// }


void exclusive_scan(int* input, int N, int* result)
{

  // cudaMemcpy(result, input, N * sizeof(int), cudaMemcpyDeviceToDevice);
  // TODO:
  //
  // Implement your exclusive scan implementation here.  Keep in
  // mind that although the arguments to this function are device
  // allocated arrays, this is a function that is running in a thread
  // on the CPU.  Your implementation will need to make multiple calls
  // to CUDA kernel functions (that you must write) to implement the
  // scan.
  // int rounded_N = nextPow2(N);
  
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

  // Debug: print initial input
  // printDeviceArray(result, N, "Initial input copied to result");

  // for(int stride = 1; stride <= N/2; stride *= 2) {
  //   int strided = 2 * stride;
  //   int num_blocks = (N / strided + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  //   upsweep<<<num_blocks, THREADS_PER_BLOCK>>>(result, N, stride);
  //   cudaDeviceSynchronize();
  //   printDeviceArray(result, N, "UPSWEEEP AFTER");
  // }

  // // Debug: print array after upsweep
  // printDeviceArray(result, N, "After upsweep");

  // int zero = 0;
  // cudaMemcpy(result + N - 1, &zero, sizeof(int), cudaMemcpyHostToDevice);

  // for(int stride = N/2; stride >= 1; stride /= 2) {
  //   int strided = 2 * stride;
  //   int num_blocks = (N / strided + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  //   downsweep<<<num_blocks, THREADS_PER_BLOCK>>>(result, N, stride);
  //   cudaDeviceSynchronize();
  // }

  
  // // Debug: print array after downsweep
  // printDeviceArray(result, N, "After downsweep");


  // int rounded_length = nextPow2(N);
  // cudaMemcpy(result, input, N * sizeof(int), cudaMemcpyDeviceToDevice);

  // int blocks = (rounded_length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  // // Debug: print initial input
  // printDeviceArray(result, N, "Initial input copied to result");

  // // Upsweep phase
  // upsweep<<<blocks, THREADS_PER_BLOCK>>>(result, rounded_length);
  // cudaDeviceSynchronize();

  // // Debug: print array after upsweep
  // printDeviceArray(result, N, "After upsweep");

  // // Downsweep phase
  // downsweep<<<blocks, THREADS_PER_BLOCK>>>(result, rounded_length);
  // cudaDeviceSynchronize();

  // // Debug: print array after downsweep
  // printDeviceArray(result, N, "After downsweep");

}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.

// Debugging function to print device arrays after copying them to host


double cudaScan(int* inarray, int* end, int* resultarray)
{
  int* device_result;
  int* device_input;
  int N = end - inarray;  

  // This code rounds the arrays provided to exclusive_scan up
  // to a power of 2, but elements after the end of the original
  // input are left uninitialized and not checked for correctness.
  //
  // Student implementations of exclusive_scan may assume an array's
  // allocated length is a power of 2 for simplicity. This will
  // result in extra work on non-power-of-2 inputs, but it's worth
  // the simplicity of a power of two only solution.

  int rounded_length = nextPow2(end - inarray);
  
  cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
  cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

  // For convenience, both the input and output vectors on the
  // device are initialized to the input values. This means that
  // students are free to implement an in-place scan on the result
  // vector if desired.  If you do this, you will need to keep this
  // in mind when calling exclusive_scan from find_repeats.
  cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

  double startTime = CycleTimer::currentSeconds();
  // printArray(device_input, N);
  // printArray(device_result, N);
  exclusive_scan(device_input, rounded_length, device_result);
  // printArray(device_result, N);

  // Wait for completion
  cudaDeviceSynchronize();
  double endTime = CycleTimer::currentSeconds();
      
  cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

  double overallDuration = endTime - startTime;
  return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
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
  // Explicitly set last element to 0
  if(threadIndex == N - 1) {
      output[threadIndex] = 0;
  }
}


__global__ void get_repeats(int* scan, int* output, int* flags, int length) {
  int threadIndex = threadIdx.x + (blockDim.x * blockIdx.x);
  // if([threadIndex] == 1)
  if(threadIndex < length - 1 && flags[threadIndex] == 1) {
    output[scan[threadIndex]] = threadIndex;
  }
}


// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {
  // TODO:
  //
  // Implement this function. You will probably want to
  // make use of one or more calls to exclusive_scan(), as well as
  // additional CUDA kernel launches.
  //    
  // Note: As in the scan code, the calling code ensures that
  // allocated arrays are a power of 2 in size, so you can use your
  // exclusive_scan function with them. However, your implementation
  // must ensure that the results of find_repeats are correct given
  // the actual array length.
  int* device_flags;
  int* device_scan;
  int rounded_length = nextPow2(length);

  cudaMalloc(&device_flags, sizeof(int) * rounded_length);
  cudaMalloc(&device_scan, sizeof(int) * rounded_length);

  // cudaMemcpy(device_input, device, N * sizeof(int), cudaMemcpyHostToDevice);
  // cudaMemcpy(device_result, inarray, N * sizeof(int), cudaMemcpyHostToDevice);
  int blocks = (length + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  map_repeats<<<blocks, THREADS_PER_BLOCK>>>(device_input, length, device_flags);
  cudaDeviceSynchronize();

  cudaMemset(device_flags + length - 1, 0, sizeof(int) * (rounded_length - length + 1));
  exclusive_scan(device_flags, rounded_length, device_scan);
  cudaDeviceSynchronize();

  int total_repeats;
  cudaMemcpy(&total_repeats, &device_scan[length - 1], sizeof(int), cudaMemcpyDeviceToHost);

  get_repeats<<<blocks, THREADS_PER_BLOCK>>>(device_scan, device_output, device_flags, length);
  cudaDeviceSynchronize();



  
  // int* 

  // exclusive_scan(device_input, length, device_output);

  // cudaDeviceSynchronize();
  // int index = 0;
  // for(int i = 0; i < length; i++) {
  //   if(device_scan[i] == 1) {
  //     device_output[index] = i;
  //     index++;
  //   }
  // }
    // Get the total count (last element of scan + last flag)
  // int total_repeats;
  // cudaMemcpy(&total_repeats, device_scan + length - 1, sizeof(int), cudaMemcpyDeviceToHost);

  
  // Clean up temporary arrays
  cudaFree(device_flags);
  cudaFree(device_scan);
  
  return total_repeats;
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
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
