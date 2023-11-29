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

__global__ void sleep_kernel(const int64_t ncycles) {
  int64_t start = clock64();
  while (clock64() - start < ncycles) {
    __syncthreads();
  }
}

extern "C" EXPORT_API void run_benchmark(double *times, const int nsamples,
                                         const int64_t ncycles) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  for (int isample = 0; isample < nsamples; ++isample) {
    auto timer = high_resolution_clock::now();
    sleep_kernel<<<1, 1, 0, stream>>>(ncycles);
    cudaStreamSynchronize(stream);
    auto elapsed = high_resolution_clock::now() - timer;
    auto time_total = duration_cast<nano_double>(elapsed).count();
    times[isample] = time_total;
  }

  cudaStreamDestroy(stream);
}
