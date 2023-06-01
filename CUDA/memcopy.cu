#include <cuda.h>
#include <stdint.h>

#include <chrono>
using namespace std::chrono;
using nano_double = duration<double, std::nano>;

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API
#endif

__global__ void memcopy_kernel(uint8_t *dst, const uint8_t *src, const int n) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix < n) {
    dst[ix] = src[ix];
  }
}

extern "C" EXPORT_API void run_benchmark(double *times, const int nsamples,
                                         const int n) {
  cudaDeviceReset();

  uint8_t *dst, *src;
  cudaMalloc(&dst, n);
  cudaMalloc(&src, n);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int nthreads = 256;
  int nblocks = (n + nthreads - 1) / nthreads;

  for (int isample = 0; isample < nsamples; ++isample) {
    auto timer = high_resolution_clock::now();
    memcopy_kernel<<<nblocks, nthreads, 0, stream>>>(dst, src, n);
    cudaStreamSynchronize(stream);
    auto elapsed = high_resolution_clock::now() - timer;
    auto time_total = duration_cast<nano_double>(elapsed).count();
    times[isample] = time_total;
  }

  cudaFree(src);
  cudaFree(dst);

  cudaStreamDestroy(stream);
}