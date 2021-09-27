#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <immintrin.h>

using namespace std::chrono;

constexpr int VEC = 16;
constexpr int LOOPS = 16;
constexpr int FMAS = LOOPS;

void peak_flops(float* in, float* out, int n) {
  __m512 one = _mm512_set1_ps(1.0f);
  __m512 y0 = one;
  __m512 y1 = one;
  __m512 y2 = one;
  __m512 y3 = one;
  __m512 y4 = one;
  __m512 y5 = one;
  __m512 y6 = one;
  __m512 y7 = one;
  for (int j = 0; j < n / VEC; j+=8) {
    __m512 x0 = reinterpret_cast<__m512*>(in)[j];
    __m512 x1 = reinterpret_cast<__m512*>(in)[j + 1];
    __m512 x2 = reinterpret_cast<__m512*>(in)[j + 2];
    __m512 x3 = reinterpret_cast<__m512*>(in)[j + 3];
    __m512 x4 = reinterpret_cast<__m512*>(in)[j + 4];
    __m512 x5 = reinterpret_cast<__m512*>(in)[j + 5];
    __m512 x6 = reinterpret_cast<__m512*>(in)[j + 6];
    __m512 x7 = reinterpret_cast<__m512*>(in)[j + 7];
    __builtin_prefetch(reinterpret_cast<void*>(&reinterpret_cast<__m512*>(in)[j + 64]));
    __builtin_prefetch(reinterpret_cast<void*>(&reinterpret_cast<__m512*>(in)[j + 64 + 1]));
    __builtin_prefetch(reinterpret_cast<void*>(&reinterpret_cast<__m512*>(in)[j + 64 + 2]));
    __builtin_prefetch(reinterpret_cast<void*>(&reinterpret_cast<__m512*>(in)[j + 64 + 3]));
    __builtin_prefetch(reinterpret_cast<void*>(&reinterpret_cast<__m512*>(in)[j + 64 + 4]));
    __builtin_prefetch(reinterpret_cast<void*>(&reinterpret_cast<__m512*>(in)[j + 64 + 5]));
    __builtin_prefetch(reinterpret_cast<void*>(&reinterpret_cast<__m512*>(in)[j + 64 + 6]));
    __builtin_prefetch(reinterpret_cast<void*>(&reinterpret_cast<__m512*>(in)[j + 64 + 7]));
    // __builtin_prefetch(reinterpret_cast<void*>(&reinterpret_cast<__m512*>(in)[j + 9]));
    // __builtin_prefetch(reinterpret_cast<void*>(&reinterpret_cast<__m512*>(in)[j + 10]));
    // __builtin_prefetch(reinterpret_cast<void*>(&reinterpret_cast<__m512*>(in)[j + 11]));
    // __builtin_prefetch(reinterpret_cast<void*>(&reinterpret_cast<__m512*>(in)[j + 12]));
    // __builtin_prefetch(reinterpret_cast<void*>(&reinterpret_cast<__m512*>(in)[j + 13]));
    // __builtin_prefetch(reinterpret_cast<void*>(&reinterpret_cast<__m512*>(in)[j + 14]));
    // __builtin_prefetch(reinterpret_cast<void*>(&reinterpret_cast<__m512*>(in)[j + 15]));
    
    // __builtin_prefetch(reinterpret_cast<void*>(&reinterpret_cast<__m512*>(out)[j]));
    // __builtin_prefetch(reinterpret_cast<void*>(&reinterpret_cast<__m512*>(in)[j + 1]));
    // __builtin_prefetch(reinterpret_cast<void*>(&reinterpret_cast<__m512*>(in)[j + 2]));
    // __builtin_prefetch(reinterpret_cast<void*>(&reinterpret_cast<__m512*>(in)[j + 3]));
    // __builtin_prefetch(reinterpret_cast<void*>(&reinterpret_cast<__m512*>(in)[j + 4]));
    // __builtin_prefetch(reinterpret_cast<void*>(&reinterpret_cast<__m512*>(in)[j + 5]));
    // __builtin_prefetch(reinterpret_cast<void*>(&reinterpret_cast<__m512*>(in)[j + 6]));
    // __builtin_prefetch(reinterpret_cast<void*>(&reinterpret_cast<__m512*>(in)[j + 7]));
    for (int i = 0; i < LOOPS; i++) {
      y0 = _mm512_fmadd_ps(y0, x0, one);
      y1 = _mm512_fmadd_ps(y1, x1, one);
      y2 = _mm512_fmadd_ps(y2, x2, one);
      y3 = _mm512_fmadd_ps(y3, x3, one);
      y4 = _mm512_fmadd_ps(y4, x4, one);
      y5 = _mm512_fmadd_ps(y5, x5, one);
      y6 = _mm512_fmadd_ps(y6, x6, one);
      y7 = _mm512_fmadd_ps(y7, x7, one);
    }
    _mm512_stream_ps(reinterpret_cast<float*>(&reinterpret_cast<__m512*>(out)[j]), y0);
    _mm512_stream_ps(reinterpret_cast<float*>(&reinterpret_cast<__m512*>(out)[j + 1]), y1);
    _mm512_stream_ps(reinterpret_cast<float*>(&reinterpret_cast<__m512*>(out)[j + 2]), y2);
    _mm512_stream_ps(reinterpret_cast<float*>(&reinterpret_cast<__m512*>(out)[j + 3]), y3);
    _mm512_stream_ps(reinterpret_cast<float*>(&reinterpret_cast<__m512*>(out)[j + 4]), y4);
    _mm512_stream_ps(reinterpret_cast<float*>(&reinterpret_cast<__m512*>(out)[j + 5]), y5);
    _mm512_stream_ps(reinterpret_cast<float*>(&reinterpret_cast<__m512*>(out)[j + 6]), y6);
    _mm512_stream_ps(reinterpret_cast<float*>(&reinterpret_cast<__m512*>(out)[j + 7]), y7);
    // reinterpret_cast<__m512*>(out)[j] = y0;
    // reinterpret_cast<__m512*>(out)[j + 1] = y1;
    // reinterpret_cast<__m512*>(out)[j + 2] = y2;
    // reinterpret_cast<__m512*>(out)[j + 3] = y3;
    // reinterpret_cast<__m512*>(out)[j + 4] = y4;
    // reinterpret_cast<__m512*>(out)[j + 5] = y5;
    // reinterpret_cast<__m512*>(out)[j + 6] = y6;
    // reinterpret_cast<__m512*>(out)[j + 7] = y7;
  }
}

void peak_flops_2(float* in, float* out, int n) {
  __m512 one = _mm512_set1_ps(1.0f);
  __m512 y0 = one;
  for (int j = 0; j < n / VEC; j++) {
    __m512 x0 = reinterpret_cast<__m512*>(in)[j];
    for (int i = 0; i < LOOPS; i++) {
      y0 = _mm512_fmadd_ps(y0, x0, one);
    }
    reinterpret_cast<__m512*>(out)[j] = y0;
  }
}

int main() {
  constexpr int64_t N = 1 << 28;
  constexpr int ITERS = 10;

  float* x = static_cast<float*>(std::aligned_alloc(sizeof(__m512), sizeof(float) * N));
  float* y = static_cast<float*>(std::aligned_alloc(sizeof(__m512), sizeof(float) * N));

  auto start = high_resolution_clock::now();
  for (int i = 0; i < ITERS; i++) {
    peak_flops(x, y, N);
  }
  auto stop = high_resolution_clock::now();

  auto millis = duration<float, std::milli>(stop - start).count();
  millis /= ITERS;
  
  float bytes = (float)N * sizeof(float) * 2;
  float flops = (float)N * FMAS * 2;

  static_assert(N * FMAS * 2 == N / VEC / 8 * LOOPS * 8 * VEC * 2);
  
  printf("%.3f ms %.1f gb/s %.1f gflops/s\n", millis, bytes / millis / 1e6, flops / millis / 1e6);

  free(x);
  free(y);
}

  
