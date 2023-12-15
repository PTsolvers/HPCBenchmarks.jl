#include "hip/hip_runtime.h"
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <time.h>

#include <chrono>
using namespace std::chrono;
using nano_double = duration<double, std::nano>;

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API
#endif

// #define DAT float
#define DAT double
// #define DAT uint32_t
// #define DAT uint64_t

#define source(i) _source[i]
#define indices(i, j) _indices[j * n + i]
#define target1(i) _target1[i]
#define target2(i) _target2[i]

__global__ void no_atomic_kernel(DAT *_target1, DAT *_target2, DAT *_source, int *_indices, const int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int i1 = indices(i, 0);
  int i2 = indices(i, 1);
  int i3 = indices(i, 2);
  int i4 = indices(i, 3);
  DAT v = source(i);
  target1(i1) += v;
  target1(i2) += v;
  target1(i3) += v;
  target1(i4) += v;
  target2(i1) += v;
  target2(i2) += v;
  target2(i3) += v;
  target2(i4) += v;
}

__global__ void atomic_kernel(DAT *_target1, DAT *_target2, DAT *_source, int *_indices, const int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int i1 = indices(i, 0);
  int i2 = indices(i, 1);
  int i3 = indices(i, 2);
  int i4 = indices(i, 3);
  DAT v = source(i);
  atomicAdd(&target1(i1), v);
  atomicAdd(&target1(i2), v);
  atomicAdd(&target1(i3), v);
  atomicAdd(&target1(i4), v);
  atomicAdd(&target2(i1), v);
  atomicAdd(&target2(i2), v);
  atomicAdd(&target2(i3), v);
  atomicAdd(&target2(i4), v);
}

extern "C" EXPORT_API void run_benchmark(double *times, const int nsamples) {
  int i;
  const int n = 1024;
  const int bins = 64;
  DAT *target1, *target2, *source;
  int *indices;

  srand((unsigned) time(NULL));

  DAT *target1_h = (DAT *)malloc(bins * sizeof(DAT));
  for (i = 0; i < bins; i++) {
    target1_h[i] = (DAT)0.0;
  }
  DAT *target2_h = (DAT *)malloc(bins * sizeof(DAT));
  for (i = 0; i < bins; i++) {
    target2_h[i] = (DAT)0.0;
  }
  DAT *source_h = (DAT *)malloc(n * sizeof(DAT));
  for (i = 0; i < n; i++) {
    source_h[i] = static_cast<DAT>(rand()) / static_cast<DAT>(RAND_MAX);
  }
  int *indices_h = (int *)malloc(n * 4 * sizeof(int));
  for (i = 0; i < (n * 4); i++) {
    indices_h[i] = std::rand() % bins;
  }

  hipMalloc(&target1, bins * sizeof(DAT));
  hipMalloc(&target2, bins * sizeof(DAT));
  hipMalloc(&source, n * sizeof(DAT));
  hipMalloc(&indices, n * 4 * sizeof(int));

  hipMemcpy(target1, target1_h, bins * sizeof(DAT), hipMemcpyHostToDevice);
  hipMemcpy(target2, target2_h, bins * sizeof(DAT), hipMemcpyHostToDevice);
  hipMemcpy(source, source_h, n * sizeof(DAT), hipMemcpyHostToDevice);
  hipMemcpy(indices, indices_h, n * 4 * sizeof(int), hipMemcpyHostToDevice);

  hipStream_t stream;
  hipStreamCreate(&stream);

  dim3 nthreads(256);
  dim3 nblocks((n + nthreads.x - 1) / nthreads.x);

  // for (int isample = 0; isample < nsamples; ++isample) {
  //   auto timer = high_resolution_clock::now();
  //   hipLaunchKernelGGL(no_atomic_kernel, nblocks, nthreads, 0, stream, target1, target2, source, indices, n);
  //   hipStreamSynchronize(stream);
  //   auto elapsed = high_resolution_clock::now() - timer;
  //   auto time_total = duration_cast<nano_double>(elapsed).count();
  //   times[isample] = time_total;
  // }

  for (int isample = 0; isample < nsamples; ++isample) {
    auto timer = high_resolution_clock::now();
    hipLaunchKernelGGL(atomic_kernel, nblocks, nthreads, 0, stream, target1, target2, source, indices, n);
    hipStreamSynchronize(stream);
    auto elapsed = high_resolution_clock::now() - timer;
    auto time_total = duration_cast<nano_double>(elapsed).count();
    times[isample] = time_total;
  }

  free(target1_h);
  free(target2_h);
  free(source_h);
  free(indices_h);
  hipFree(target1);
  hipFree(target2);
  hipFree(source);
  hipFree(indices);

  hipStreamDestroy(stream);
}
