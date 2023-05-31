#include <cuda.h>

#include <chrono>
using nano_double = std::chrono::duration<double, std::nano>;

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API
#endif

__global__ void busy_kernel(const int nspins) {
  int ispin = 0;
  while (ispin < nspins) {
    ispin += 1;
    __syncthreads();
  }
  return;
}

extern "C" EXPORT_API void run_benchmark(double *times, const int nsamples,
                                         const int nspins) {
  cudaDeviceReset();

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  for (int isample = 0; isample < nsamples; ++isample) {
    auto t1 = std::chrono::high_resolution_clock::now();
    busy_kernel<<<1, 1, 0, stream>>>(nspins);
    cudaStreamSynchronize(stream);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto time_total = std::chrono::duration_cast<nano_double>(t2 - t1).count();
    times[isample] = time_total;
  }

  cudaStreamDestroy(stream);
}