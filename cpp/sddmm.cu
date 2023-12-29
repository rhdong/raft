#include <chrono>
#include <cuda_runtime_api.h>  // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>          // cusparseSpMM
#include <random>
#include <stdio.h>   // printf
#include <stdlib.h>  // EXIT_FAILURE

#define CHECK_CUDA(func)                                         \
  {                                                              \
    cudaError_t status = (func);                                 \
    if (status != cudaSuccess) {                                 \
      printf("CUDA API failed at line %d with error: %s (%d)\n", \
             __LINE__,                                           \
             cudaGetErrorString(status),                         \
             status);                                            \
      return EXIT_FAILURE;                                       \
    }                                                            \
  }

#define CHECK_CUSPARSE(func)                                         \
  {                                                                  \
    cusparseStatus_t status = (func);                                \
    if (status != CUSPARSE_STATUS_SUCCESS) {                         \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
             __LINE__,                                               \
             cusparseGetErrorString(status),                         \
             status);                                                \
      return EXIT_FAILURE;                                           \
    }                                                                \
  }

template <typename Rep>
struct Timer {
  explicit Timer(TimeUnit tu = TimeUnit::Second) : tu_(tu) {}
  void start() { startRecord = std::chrono::steady_clock::now(); }
  void end() { endRecord = std::chrono::steady_clock::now(); }
  Rep getResult()
  {
    auto duration_ = std::chrono::duration_cast<std::chrono::nanoseconds>(endRecord - startRecord);
    auto pow_      = static_cast<int32_t>(tu_) - static_cast<int32_t>(TimeUnit::NanoSecond);
    auto factor    = static_cast<Rep>(std::pow(10, pow_));
    return static_cast<Rep>(duration_.count()) * factor;
  }

 private:
  TimeUnit tu_;
  std::chrono::time_point<std::chrono::steady_clock> startRecord{};
  std::chrono::time_point<std::chrono::steady_clock> endRecord{};
};

struct SDDMMBenchParams {
  size_t m;  // m parameter of the SDDMM
  size_t k;  // k parameter of the SDDMM
  size_t n;  // n parameter of the SDDMM
  float sparsity;
  ValueType alpha = 1.0;
  ValueType beta  = 0.0;
};

void convert_to_csr(std::vector<bool>& matrix,
                    size_t rows,
                    size_t cols,
                    std::vector<ValueType>& values,
                    std::vector<IndexType>& indices,
                    std::vector<IndexType>& indptr)
{
  indptr.push_back(0);

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      if (matrix[i * cols + j]) {
        values.push_back(static_cast<ValueType>(1.0f));
        indices.push_back(static_cast<IndexType>(j));
      }
    }
    indptr.push_back(static_cast<IndexType>(values.size()));
  }
}

size_t create_sparse_matrix(size_t m, size_t n, float sparsity, bool* matrix)
{
  size_t total_elements = static_cast<size_t>(m * n);
  size_t num_ones       = static_cast<size_t>((total_elements * 1.0f) * sparsity);
  size_t res            = num_ones;

  for (size_t i = 0; i < total_elements; ++i) {
    matrix[i] = false;
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, total_elements - 1);

  while (num_ones > 0) {
    size_t index = dis(gen);

    if (matrix[index] == false) {
      matrix[index] = true;
      num_ones--;
    }
  }
  return res;
}

void uniform(float* array, int size)
{
  std::random_device rd;
  std::mt19937 eng(rd());
  std::uniform_real_distribution<> distr(0.0, 1.0);

  std::vector<float> randomArray(arraySize);

  for (int i = 0; i < size; ++i) {
    randomArray[i] = static_cast<float>(distr(eng));
  }
}

void test_main(SDDMMBenchParams& params, Timer<double>& timer)
{
  // Host problem definition
  int A_num_rows = params.m;
  int A_num_cols = params.k;
  int B_num_rows = A_num_cols;
  int B_num_cols = params.n;
  int lda        = A_num_cols;
  int ldb        = B_num_cols;
  int A_size     = lda * A_num_rows;
  int B_size     = ldb * B_num_rows;
  int C_size     = A_num_rows * B_num_cols;
  float hA*      = malloc(sizeof(float) * A_size);
  float hB*      = malloc(sizeof(float) * B_size);

  uniform(hA, A_size);
  uniform(hB, B_size);

  float c_dense_data_h* = malloc(sizeof(bool) * C_size);

  size_t c_true_nnz = create_sparse_matrix(A_num_rows, B_num_cols, params.sparsity, c_dense_data_h);

  int* hC_offsets  = malloc(sizeof(int) * (params.m + 1));
  int* hC_columns  = malloc(sizeof(int) * c_true_nnz);
  float* hC_values = malloc(sizeof(float) * c_true_nnz);

  convert_to_csr(c_dense_data_h, params.m, params.n, hC_values, hC_columns, hC_offsets);
  //--------------------------------------------------------------------------
  // Device memory management
  int *dC_offsets, *dC_columns;
  float *dC_values, *dB, *dA;
  CHECK_CUDA(cudaMalloc((void**)&dA, A_size * sizeof(float)))
  CHECK_CUDA(cudaMalloc((void**)&dB, B_size * sizeof(float)))
  CHECK_CUDA(cudaMalloc((void**)&dC_offsets, (A_num_rows + 1) * sizeof(int)))
  CHECK_CUDA(cudaMalloc((void**)&dC_columns, c_true_nnz * sizeof(int)))
  CHECK_CUDA(cudaMalloc((void**)&dC_values, c_true_nnz * sizeof(float)))

  CHECK_CUDA(cudaMemcpy(dA, hA, A_size * sizeof(float), cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dB, hB, B_size * sizeof(float), cudaMemcpyHostToDevice))
  CHECK_CUDA(
    cudaMemcpy(dC_offsets, hC_offsets, (A_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dC_columns, hC_columns, c_true_nnz * sizeof(int), cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dC_values, hC_values, c_true_nnz * sizeof(float), cudaMemcpyHostToDevice))
  //--------------------------------------------------------------------------
  // CUSPARSE APIs
  cusparseHandle_t handle = NULL;
  cusparseDnMatDescr_t matA, matB;
  cusparseSpMatDescr_t matC;
  void* dBuffer     = NULL;
  size_t bufferSize = 0;
  CHECK_CUSPARSE(cusparseCreate(&handle))
  // Create dense matrix A
  CHECK_CUSPARSE(
    cusparseCreateDnMat(&matA, A_num_rows, A_num_cols, lda, dA, CUDA_R_32F, CUSPARSE_ORDER_ROW))
  // Create dense matrix B
  CHECK_CUSPARSE(
    cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB, CUDA_R_32F, CUSPARSE_ORDER_COL))
  // Create sparse matrix C in CSR format
  CHECK_CUSPARSE(cusparseCreateCsr(&matC,
                                   A_num_rows,
                                   B_num_cols,
                                   c_true_nnz,
                                   dC_offsets,
                                   dC_columns,
                                   dC_values,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_32F))
  // allocate an external buffer if needed
  CHECK_CUSPARSE(cusparseSDDMM_bufferSize(handle,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &params.alpha,
                                          matA,
                                          matB,
                                          &params.beta,
                                          matC,
                                          CUDA_R_32F,
                                          CUSPARSE_SDDMM_ALG_DEFAULT,
                                          &bufferSize))
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

  // execute preprocess (optional)
  //   CHECK_CUSPARSE(cusparseSDDMM_preprocess(handle,
  //                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
  //                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
  //                                           &alpha,
  //                                           matA,
  //                                           matB,
  //                                           &beta,
  //                                           matC,
  //                                           CUDA_R_32F,
  //                                           CUSPARSE_SDDMM_ALG_DEFAULT,
  //                                           dBuffer))
  // execute SpMM
  cudaStream_t stream;

  CHECK_CUDA(cudaStreamCreate(stream));
  timer.start();
  CHECK_CUSPARSE(cusparseSDDMM(handle,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha,
                               matA,
                               matB,
                               &beta,
                               matC,
                               CUDA_R_32F,
                               CUSPARSE_SDDMM_ALG_DEFAULT,
                               dBuffer,
                               stream))

  CHECK_CUDA(cudaStreamSynchronize(stream))
  timer.end();
  CHECK_CUDA(cudaStreamDestroy(stream));

  // destroy matrix/vector descriptors
  CHECK_CUSPARSE(cusparseDestroyDnMat(matA))
  CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
  CHECK_CUSPARSE(cusparseDestroySpMat(matC))
  CHECK_CUSPARSE(cusparseDestroy(handle))

  //--------------------------------------------------------------------------
  // device memory deallocation
  CHECK_CUDA(cudaFree(dBuffer))
  CHECK_CUDA(cudaFree(dA))
  CHECK_CUDA(cudaFree(dB))
  CHECK_CUDA(cudaFree(dC_offsets))
  CHECK_CUDA(cudaFree(dC_columns))
  CHECK_CUDA(cudaFree(dC_values))

  free(hA);
  free(hB);
  free(c_dense_data_h);
  free(hC_offsets);
  free(hC_columns);
  free(hC_values);
}

int main(void)
{
  std::vector<SDDMMBenchParams> cases{{1024 * 1024, 128, 1024, 0.249, 1.0f, 0.0f},
                                      {1024 * 1024, 128, 1024, 0.251, 1.0f, 0.0f}};

  auto timer = Timer<double>();
  for (auto params : cases) {
    test_main(params, timer);
  }
  return EXIT_SUCCESS;
}