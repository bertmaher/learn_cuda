#include <stdio.h>

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorString(result), func);
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template<typename T, typename U>
constexpr T ceildiv(T t, U u) {
  return (t + u - 1) / u;
}

constexpr int LOOPS_1 = 32;
constexpr int FMAS = LOOPS_1;

__global__
void peak_flops(float* in, float* out, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    float x = in[i];
    float y = 1.0f;
    for (int j = 0; j < LOOPS_1; j++) {
      y = fma(y, x, 1.0f);
    }
    out[i] = y;
  }
}

constexpr int LOOPS_2 = 256;
constexpr int FMAS_2 = LOOPS_2 * 4;

__global__
void peak_flops_2(float* in, float* out, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    float x = in[i];
    float y = 1.0f;
    float z = -1.0f;
    float w = 2.0f;
    float u = -2.0f;
    for (int j = 0; j < LOOPS_2; j++) {
      y = fma(y, x, 1.0f);
      z = fma(z, x, -1.0f);
      w = fma(w, x, 2.0f);
      u = fma(u, x, -2.0f);
    }
    out[i] = y + z + w + u;
  }
}

void init(float* in, int n) {
  for (int i = 0; i < n; i++) {
    in[i] = (float)(i) / n;
  }
}

int main() {
  constexpr int N = (1 << 28);
  constexpr int ITERS = 100;
  
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N * sizeof(float));
  y = (float*)malloc(N * sizeof(float));
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_y, N * sizeof(float)));

  init(x, N);
  
  checkCudaErrors(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  
  checkCudaErrors(cudaEventRecord(start));
  int blockSize = 32;
  for (int i = 0; i < ITERS; i++) {
    peak_flops<<<ceildiv(N, blockSize), blockSize>>>(d_x, d_y, N);
  }
  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  float millis = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&millis, start, stop));
  millis /= ITERS;
  
  float bytes = (float)N * sizeof(float) * 2;
  float flops = (float)N * FMAS * 2;

  printf("%.3f ms %.1f gb/s %.1f gflops/s\n", millis, bytes / millis / 1e6, flops / millis / 1e6);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}
