// This program computes a matrix multiplication using cuBLAS sgemm
// By: Antigravity (derived from CoffeeBeforeArch concepts)

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Check result on the CPU
void verify_result(const std::vector<float> &a, const std::vector<float> &b,
                   const std::vector<float> &c, int M, int N, int K,
                   float alpha, float beta) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float tmp = 0.0f;
      for (int k = 0; k < K; k++) {
        tmp += a[i * K + k] * b[k * N + j];
      }
      // Since C is initialized to random (but beta is 0.0f in our call), 
      // the expected result is alpha * tmp. If beta were non-zero, we would add beta * c_orig.
      float expected = alpha * tmp;
      assert(std::abs(c[i * N + j] - expected) < 1e-2);
    }
  }
}

int main() {
  // Matrix dimensions (M x K) * (K x N) = (M x N)
  const int M = 1 << 10; // 1024
  const int N = 1 << 10; // 1024
  const int K = 1 << 10; // 1024

  size_t bytes_a = M * K * sizeof(float);
  size_t bytes_b = K * N * sizeof(float);
  size_t bytes_c = M * N * sizeof(float);

  // Host vectors (Row-major)
  std::vector<float> h_a(M * K);
  std::vector<float> h_b(K * N);
  std::vector<float> h_c(M * N);

  // Initialize matrices with float values
  std::generate(h_a.begin(), h_a.end(), []() { return static_cast<float>(rand() % 10); });
  std::generate(h_b.begin(), h_b.end(), []() { return static_cast<float>(rand() % 10); });

  // Allocate device memory
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes_a);
  cudaMalloc(&d_b, bytes_b);
  cudaMalloc(&d_c, bytes_c);

  // Copy data to the device
  cudaMemcpy(d_a, h_a.data(), bytes_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes_b, cudaMemcpyHostToDevice);

  // Create cuBLAS handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Parameters for sgemm: C = alpha * A * B + beta * C
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Perform Matrix Multiplication using cuBLAS sgemm
  // In C++ (Row-Major), we compute: C_rm = A_rm * B_rm
  // In cuBLAS (Column-Major), this is equivalent to: C_cm^T = B_cm^T * A_cm^T
  // Where B_cm^T is N x K, A_cm^T is K x M, yielding C_cm^T of N x M.
  // Therefore, we pass transa = CUBLAS_OP_N, transb = CUBLAS_OP_N
  // m = N, n = M, k = K
  // leading dimension of A (which is B^T in col-major) lda = N
  // leading dimension of B (which is A^T in col-major) ldb = K
  // leading dimension of C (which is C^T in col-major) ldc = N
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N);

  // Copy back to host
  cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost);

  // Check result
  verify_result(h_a, h_b, h_c, M, N, K, alpha, beta);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  // Free resources
  cublasDestroy(handle);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
