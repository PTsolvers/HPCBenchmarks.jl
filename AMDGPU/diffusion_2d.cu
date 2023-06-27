#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <stdint.h>

#include <chrono>
using namespace std::chrono;
using nano_double = duration<double, std::nano>;

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API
#endif

#define A_new(ix, iy) _A_new[(iy) * (n - 2) + ix]
#define A(ix, iy) _A[(iy)*n + ix]

__global__ void diffusion_kernel(double *_A_new, const double *_A, const int n, const double h) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ix < n - 2 && iy < n - 2) {
    A_new(ix, iy) = A(ix + 1, iy + 1) + h * (A(ix, iy + 1) + A(ix + 2, iy + 1) + A(ix + 1, iy) + A(ix + 1, iy + 2) -
                                             4.0 * A(ix + 1, iy + 1));
  }
}

extern "C" EXPORT_API void run_benchmark(double *times, const int nsamples, const int n) {
  double *A_new, *A;
  hipMalloc(&A_new, (n - 2) * (n - 2) * sizeof(double));
  hipMalloc(&A, n * n * sizeof(double));

  double h = 1.0 / 5.0;

  hipStream_t stream;
  hipStreamCreate(&stream);

  dim3 nthreads(128, 2);
  dim3 nblocks((n + nthreads.x - 1) / nthreads.x, (n + nthreads.y - 1) / nthreads.y);

  for (int isample = 0; isample < nsamples; ++isample) {
    auto timer = high_resolution_clock::now();
    hipLaunchKernelGGL(diffusion_kernel, nblocks, nthreads, 0, stream, A_new, A, n, h);
    hipStreamSynchronize(stream);
    auto elapsed = high_resolution_clock::now() - timer;
    auto time_total = duration_cast<nano_double>(elapsed).count();
    times[isample] = time_total;
  }

  hipFree(A_new);
  hipFree(A);

  hipStreamDestroy(stream);
}