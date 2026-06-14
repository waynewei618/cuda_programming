// This program computes the sum of two vectors of length N using cuBLAS saxpy
// By: Antigravity (derived from CoffeeBeforeArch concepts)

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Check vector add result
void verify_result(const std::vector<float> &a, const std::vector<float> &b,
                   const std::vector<float> &c, float alpha) {
  for (size_t i = 0; i < a.size(); i++) {
    // Check with a small tolerance due to floating point precision
    assert(std::abs(c[i] - (alpha * a[i] + b[i])) < 1e-4);
  }
}

int main() {
  // Array size of 2^16 (65536 elements)
  constexpr int N = 1 << 16;
  constexpr size_t bytes = sizeof(float) * N;

  // Vectors for holding the host-side (CPU-side) data
  std::vector<float> h_x(N);
  std::vector<float> h_y(N);
  std::vector<float> h_y_orig(N); // Store original Y to verify
  std::vector<float> h_y_res(N);  // Store result Y

  // Initialize random numbers in each array
  for (int i = 0; i < N; i++) {
    h_x[i] = static_cast<float>(rand() % 100);
    h_y[i] = static_cast<float>(rand() % 100);
    h_y_orig[i] = h_y[i];
  }

  // Allocate memory on the device
  float *d_x, *d_y;
  cudaMalloc(&d_x, bytes);
  cudaMalloc(&d_y, bytes);

  // Copy data from host to device (CPU -> GPU)
  cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice);

  // Create cuBLAS handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Set alpha
  const float alpha = 2.0f;

  // Call cublasSaxpy (y = alpha * x + y)
  // handle: cuBLAS context
  // n: vector length
  // &alpha: scale factor
  // d_x: pointer to x
  // incx: stride of x
  // d_y: pointer to y
  // incy: stride of y
  cublasSaxpy(handle, N, &alpha, d_x, 1, d_y, 1);

  // Copy result vector from device to host
  cudaMemcpy(h_y_res.data(), d_y, bytes, cudaMemcpyDeviceToHost);

  // Check result for errors
  verify_result(h_x, h_y_orig, h_y_res, alpha);

  // Clean up
  cublasDestroy(handle);
  cudaFree(d_x);
  cudaFree(d_y);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}
