#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <cmath>

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

template<typename Fn>
inline __host__ __device__ float4 apply(Fn&& fn, float4 in) {
  float4 out;
  out.x = fn(in.x);
  out.y = fn(in.y);
  out.z = fn(in.z);
  out.w = fn(in.w);
  return out;
}

inline __host__ __device__ float4 tanh(float4 in) {
  return apply(tanhf, in);
}

inline __host__ __device__ float4 log1p(float4 in) {
  return apply(log1pf, in);
}

inline __host__ __device__ float4 exp(float4 in) {
  return apply(expf, in);
}

inline __host__ __device__ float4 operator*(float4 a, float4 b) {
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__global__
void mish_oneThreadPerElt(float* d_in, float* d_out, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    float f = d_in[i];
    d_out[i] = f * tanh(log1p(exp(f)));
  }
}

void launch_oneThreadPerElt(float* d_in, float* d_out, int n, int threadsPerBlock = 1024) {
  int blocks = ceildiv(n, threadsPerBlock);
  mish_oneThreadPerElt<<<blocks, threadsPerBlock>>>(d_in, d_out, n);
}

__global__
void mish_gridStride(float* d_in, float* d_out, int n) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    float f = d_in[i];
    d_out[i] = f * tanh(log1p(exp(f)));
  }
}

void launch_gridStride(float* d_in, float* d_out, int n, int blocks, int threads) {
  mish_gridStride<<<blocks, threads>>>(d_in, d_out, n);
}

__global__
void mish_oneThreadPerElt_vectorized(float* d_in, float* d_out, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n / 4) {
    float4 f = reinterpret_cast<float4*>(d_in)[i];
    reinterpret_cast<float4*>(d_out)[i] = f * tanh(log1p(exp(f)));
  }
}

void launch_oneThreadPerElt_vectorized(float* d_in, float* d_out, int n) {
  int threads = 512;
  int blocks = ceildiv(n / 4, threads);
  mish_oneThreadPerElt_vectorized<<<blocks, threads>>>(d_in, d_out, n);
}

__global__
void mish_gridStride_vectorized(float* d_in, float* d_out, int n) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n / 4; i += gridDim.x * blockDim.x) {
    float4 f = reinterpret_cast<float4*>(d_in)[i];
    reinterpret_cast<float4*>(d_out)[i] = f * tanh(log1p(exp(f)));
  }
}

void launch_gridStride_vectorized(float* d_in, float* d_out, int n, int blocks, int threads) {
  mish_gridStride_vectorized<<<blocks, threads>>>(d_in, d_out, n);
}

__global__
void mish_onethread_vec_unroll(float* d_in, float* d_out, int n) {
  int i = 8 * (blockDim.x * blockIdx.x + threadIdx.x);
  if (i < n) {
    float4 f1 = *reinterpret_cast<float4*>(d_in + i);
    float4 f2 = *reinterpret_cast<float4*>(d_in + i + 4);
    *reinterpret_cast<float4*>(d_out + i) = f1 * tanh(log1p(exp(f1)));
    *reinterpret_cast<float4*>(d_out + i + 4) = f2 * tanh(log1p(exp(f2)));
  }
}

void launch_onethread_vec_unroll(float* d_in, float* d_out, int n) {
  int blocks = ceildiv(n / 8, 1024);
  mish_onethread_vec_unroll<<<blocks, 1024>>>(d_in, d_out, n);
}

__global__
void mish_onethread_vec_unroll2(float* d_in, float* d_out, int n) {
  int i = 8 * blockDim.x * blockIdx.x;
  int ti = threadIdx.x;
  int idx = i + 4 * ti;
  if (i < n) {
    float4 f1 = *reinterpret_cast<float4*>(d_in + idx);
    *reinterpret_cast<float4*>(d_out + idx) = f1 * tanh(log1p(exp(f1)));
    float4 f2 = *reinterpret_cast<float4*>(d_in + idx + 4 * blockDim.x);
    *reinterpret_cast<float4*>(d_out + idx + 4 * blockDim.x) = f2 * tanh(log1p(exp(f2)));
  }
}

void launch_onethread_vec_unroll2(float* d_in, float* d_out, int n) {
  int blocks = ceildiv(n / 8, 1024);
  mish_onethread_vec_unroll2<<<blocks, 1024>>>(d_in, d_out, n);
}

struct float8 {
  float f0;
  float f1;
  float f2;
  float f3;
  float f4;
  float f5;
  float f6;
  float f7;
};

__global__
void mish_onethread_vec8(float* d_in, float* d_out, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n / 8) {
    float8 f1 = reinterpret_cast<float8*>(d_in)[i];
    reinterpret_cast<float8*>(d_out)[i] = f1; //* tanh(log1p(exp(f1)));
  }
}

void launch_onethread_vec8(float* d_in, float* d_out, int n) {
  int blocks = ceildiv(n / 8, 1024);
  mish_onethread_vec8<<<blocks, 1024>>>(d_in, d_out, n);
}

__global__
void mish_contiguous_vectorized(float* d_in, float* d_out, int n) {
  int chunkSize = n / 4 / gridDim.x;
  
  for (int i = blockIdx.x * chunkSize + threadIdx.x; i < (chunkSize * (blockIdx.x + 1)); i += blockDim.x) {
    float4 f = reinterpret_cast<float4*>(d_in)[i];
    reinterpret_cast<float4*>(d_out)[i] = f * tanh(log1p(exp(f)));
  }
}

void launch_contiguous_vectorized(float* d_in, float* d_out, int n, int blocks, int threads) {
  mish_contiguous_vectorized<<<blocks, threads>>>(d_in, d_out, n);
}

void init(float* in, int n) {
  for (int i = 0; i < n; i++) {
    in[i] = (float)(i) / n;
  }
}

void checkResult(float* in, float* out, int n) {
  float maxError = 0.0f;
  int errors = 0;
  for (int i = 0; i < n; i++) {
    float f = in[i];
    float ref = f * tanh(log1p(exp(f)));
    float err = std::abs(ref - out[i]);
    //std::cout << "ref: " << ref << " out: " << out[i] << "\n";
    if (err > 1e-6) {
      //std::cout << "idx: " << i << " ref: " << ref << " out: " << out[i] << "\n";
      errors++;
    }
    maxError = std::max(maxError, err);
  }
  if (maxError > 1e-6) {
    std::cout << errors << " errors\n";
    std::cout << "maxError: " << maxError << "\n";

    exit(EXIT_FAILURE);
  }
}

template <typename Fn>
void benchmark(Fn&& fn, float* d_in, float* d_out, float* h_in, float* h_out, int n) {
  constexpr int ITERS = 100;

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  float total = 0.0f;
  for (int i = 0; i < ITERS; i++) {
    checkCudaErrors(cudaEventRecord(start));
    fn(d_in, d_out, n);
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    float millis = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&millis, start, stop));
    total += millis;
  }
  
  
  checkCudaErrors(cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  checkResult(h_in, h_out, n);

  float bytes = n * sizeof(float) * 2;
  std::cout << "Time per kernel: " << total / ITERS << " ms\n";
  std::cout << "Effective bandwidth: " << bytes * ITERS / total / 1e6 << " gb/s\n";
  std::cout << "\n";
}

int main() {
  constexpr int N = (1 << 28);
  
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N * sizeof(float));
  y = (float*)malloc(N * sizeof(float));
  checkCudaErrors(cudaMalloc(&d_x, N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_y, N * sizeof(float)));

  init(x, N);
  
  checkCudaErrors(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice));

  /*
  for (int threadsPerBlock = 32; threadsPerBlock <= 1024; threadsPerBlock <<= 1) {
    std::cout << "One thread per elt, " << threadsPerBlock << " threads per block\n";
    benchmark([&](float* d_in, float* d_out, int n) {
		launch_oneThreadPerElt(d_in, d_out, n, threadsPerBlock);
	      },
	      d_x, d_y, x, y, N);
  }

  for (int blocks = 32; blocks <= ceildiv(N, 1024); blocks <<= 1) {
    std::cout << "Grid stride loop, " << blocks << " blocks, 1024 threads\n";
    benchmark([&](float* d_in, float* d_out, int n) {
	launch_gridStride(d_in, d_out, n, blocks, 1024);
      },
      d_x, d_y, x, y, N);
  }

  std::cout << "One thread per elt, vectorized\n";
  benchmark(launch_oneThreadPerElt_vectorized, d_x, d_y, x, y, N);
  
  for (int blocks = 32; blocks <= ceildiv(N / 4, 1024); blocks <<= 1) {
    std::cout << "Grid stride loop, " << blocks << " blocks, 1024 threads, vectorized\n";
    benchmark([&](float* d_in, float* d_out, int n) {
	launch_gridStride_vectorized(d_in, d_out, n, blocks, 1024);
      },
      d_x, d_y, x, y, N);
  }

  checkCudaErrors(cudaMemset(d_y, 0, N * sizeof(float)));
  std::cout << "Contiguous blocks: 512 blocks by 1024 threads\n";
    benchmark([&](float* d_in, float* d_out, int n) {
	launch_contiguous_vectorized(d_in, d_out, n, 512, 1024);
      },
      d_x, d_y, x, y, N);
  */

   benchmark(launch_oneThreadPerElt_vectorized, d_x, d_y, x, y, N);
  // benchmark(launch_onethread_vec_unroll, d_x, d_y, x, y, N);
  //benchmark(launch_onethread_vec_unroll2, d_x, d_y, x, y, N);
  //benchmark(launch_onethread_vec8, d_x, d_y, x, y, N);
 
  cudaDeviceSynchronize();
  
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}
