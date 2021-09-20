#include <stdio.h>
#include <algorithm>
#include <cmath>

__global__
void mish(int n, float* tx, float* aten_mul) {
  for (int i = (threadIdx.x + blockDim.x * blockIdx.x) * 4; i < n; i += gridDim.x * blockDim.x * 4) {
    float4 tx4 = __ldg(reinterpret_cast<float4*>(tx + i));

    tx4.x = tx4.x * tanh(log1p(exp(tx4.x)));
    tx4.y = tx4.y * tanh(log1p(exp(tx4.y)));
    tx4.z = tx4.z * tanh(log1p(exp(tx4.z)));
    tx4.w = tx4.w * tanh(log1p(exp(tx4.w)));

    *reinterpret_cast<float4*>(aten_mul + i) = tx4;
  }
}
template<typename T, typename U>
constexpr T ceildiv(T t, U u) {
  return (t + u - 1) / u;
}

int main() {
  constexpr int N = 1 << 28;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N * sizeof(float));
  y = (float*)malloc(N * sizeof(float));
  cudaMalloc(&d_x, N * sizeof(float));
  cudaMalloc(&d_y, N * sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 3.0f;
    y[i] = 2.0f;
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

  // constexpr int blockSize = 512;
  // constexpr int nBlocks = ceildiv(N, blockSize);
  constexpr int blockSize = 512;
  constexpr int nBlocks = ceildiv(N, blockSize) / 4;
  float millis = 0.0f;
  float temp = 0.0f;
  for (int i = 0; i < 500; i++) {
    cudaEventRecord(start);
    mish<<<nBlocks, blockSize>>>(N, d_x, d_y);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&temp, start, stop);
    millis += temp;
  }
  millis = millis / 500;

  cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
  
  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    float mv = 3.0f * tanhf(std::log1p(std::exp(3.0)));
    maxError = std::max(maxError, std::abs(mv - y[i]));
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
