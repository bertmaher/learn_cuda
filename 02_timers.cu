//
// From: https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
// by Mark Harris
//

#include <algorithm>
#include <cmath>
#include <stdio.h>

template<typename T, typename U>
constexpr T ceildiv(T t, U u) {
  return (t + u - 1) / u;
}

__global__
void saxpy(int n, float a, float* x, float* y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}

int main() {
  constexpr int N = 1 << 20;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N * sizeof(float));
  y = (float*)malloc(N * sizeof(float));
  cudaMalloc(&d_x, N * sizeof(float));
  cudaMalloc(&d_y, N * sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

  constexpr int blockSize = 256;
  constexpr int nBlocks = ceildiv(N, blockSize);
  cudaEventRecord(start);
  saxpy<<<nBlocks, blockSize>>>(N, 2.0f, d_x, d_y);
  cudaEventRecord(stop);

  cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);
  float millis = 0.0f;
  cudaEventElapsedTime(&millis, start, stop);
  
  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = std::max(maxError, std::abs(y[i] - 4.0f));
  }
  printf("max error: %f\n", maxError);
  printf("duration (ms): %f\n", millis);
  printf("effective bandwidth (gb/s): %f\n", (float)N * sizeof(float) * 3 / millis / 1e6);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);

  return 0;
}

  
