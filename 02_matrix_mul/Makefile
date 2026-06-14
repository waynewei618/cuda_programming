# Makefile for compiling cuBLAS examples
NVCC = nvcc
NVCC_FLAGS = -g -G -lcublas

all: cublas_vector_add cublas_matrix_mul

cublas_vector_add: cublas_vector_add/cublas_vector_add.cu
	$(NVCC) $(NVCC_FLAGS) $< -o cublas_vector_add/cublas_vector_add.out

cublas_matrix_mul: cublas_matrix_mul/cublas_matrix_mul.cu
	$(NVCC) $(NVCC_FLAGS) $< -o cublas_matrix_mul/cublas_matrix_mul.out

clean:
	rm -f cublas_vector_add/cublas_vector_add.out cublas_matrix_mul/cublas_matrix_mul.out
