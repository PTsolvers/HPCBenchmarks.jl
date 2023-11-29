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

#define A_new(ix, iy, iz) _A_new[(iz) * (n - 2) * (n - 2) + (iy) * (n - 2) + ix]
#define A(ix, iy, iz) _A[(iz) * (n - 2) * (n - 2) + (iy)*n + ix]

__global__ void diffusion_kernel(double *_A_new, const double *_A, const int n, const double h) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  if (ix < n - 2 && iy < n - 2 && iz < n - 2) {
    A_new(ix, iy, iz) = A(ix + 1, iy + 1, iz + 1) + h * (A(ix, iy + 1, iz + 1) + A(ix + 2, iy + 1, iz + 1)
                                                       + A(ix + 1, iy, iz + 1) + A(ix + 1, iy + 2, iz + 1) -
                                                       + A(ix + 1, iy + 1, iz) + A(ix + 1, iy + 1, iz + 2) - 6.0 * A(ix + 1, iy + 1, iz + 1));
  }
}

extern "C" EXPORT_API void run_benchmark(double *times, const int nsamples, const int n) {
  double *A_new, *A;
  cudaMalloc(&A_new, (n - 2) * (n - 2) * (n - 2) * sizeof(double));
  cudaMalloc(&A, n * n * n * sizeof(double));

  double h = 1.0 / 7.0;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  dim3 nthreads(32, 8, 1);
  dim3 nblocks((n + nthreads.x - 1) / nthreads.x, (n + nthreads.y - 1) / nthreads.y, (n + nthreads.z - 1) / nthreads.z);

  for (int isample = 0; isample < nsamples; ++isample) {
    auto timer = high_resolution_clock::now();
    diffusion_kernel<<<nblocks, nthreads, 0, stream>>>(A_new, A, n, h);
    cudaStreamSynchronize(stream);
    auto elapsed = high_resolution_clock::now() - timer;
    auto time_total = duration_cast<nano_double>(elapsed).count();
    times[isample] = time_total;
  }

  cudaFree(A_new);
  cudaFree(A);

  cudaStreamDestroy(stream);
}
