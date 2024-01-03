#include <chrono>
#include <cuda_runtime_api.h>  // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>          // cusparseSpMM
#include <iostream>
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
      return;                                                    \
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
      return;                                                        \
    }                                                                \
  }
enum class TimeUnit {
  Second      = 0,
  MilliSecond = 3,
  MicroSecond = 6,
  NanoSecond  = 9,
};

template <typename Rep>
struct Timer {
  explicit Timer(TimeUnit tu = TimeUnit::MilliSecond) : tu_(tu) {}
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
  float alpha = 1.0;
  float beta  = 0.0;
};

template <typename ValueType, typename IndexType = int64_t>
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

size_t create_sparse_matrix(size_t m, size_t n, float sparsity, std::vector<bool>& matrix)
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
  std::mt19937 eng(123456ULL);
  std::uniform_real_distribution<> distr(-1.0f, 1.0f);

  std::vector<float> randomArray(size);

  for (int i = 0; i < size; ++i) {
    randomArray[i] = static_cast<float>(distr(eng));
  }
}

template <typename ValueType, typename IndexType = int64_t>
void convert_to_csr_problem(std::vector<bool>& matrix,
                    size_t rows,
                    size_t cols,
                    float* values,
                    IndexType* indices,
                    IndexType* indptr)
{
  IndexType offset_indptr = 0;
  IndexType offset_values = 0;
  indptr[offset_indptr++] = 0;

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      if (matrix[i * cols + j]) {
        values[offset_values]  = static_cast<float>(1.0f);
        indices[offset_values] = static_cast<IndexType>(j);
        offset_values++;
      }
    }
    indptr[offset_indptr++] = static_cast<IndexType>(offset_values);
  }
}

void test_main(SDDMMBenchParams& params, Timer<double>& timer)
{
  // Host problem definition
  size_t lda    = params.k;
  size_t ldb    = params.k;
  size_t A_size = params.m * params.k;
  size_t B_size = params.k * params.n;
  size_t C_size = params.m * params.n;
  float* hA     = (float*)malloc(sizeof(float) * A_size);
  float* hB     = (float*)malloc(sizeof(float) * B_size);

  uniform(hA, A_size);
  uniform(hB, B_size);

  std::vector<bool> c_dense_data_h(C_size);

  size_t c_true_nnz = create_sparse_matrix(params.m, params.n, params.sparsity, c_dense_data_h);

  std::cout << "c_true_nnz: " << c_true_nnz << std::endl;

  std::vector<float> hC_values(c_true_nnz);
  std::vector<int64_t> hC_columns(c_true_nnz);
  std::vector<int64_t> hC_offsets(params.m + 1);

  convert_to_csr<float, int64_t>(
    c_dense_data_h, params.m, params.n, hC_values, hC_columns, hC_offsets);
  //--------------------------------------------------------------------------
  // Device memory management
  int64_t *dC_offsets, *dC_columns;
  float *dC_values, *dB, *dA;
  CHECK_CUDA(cudaMalloc((void**)&dA, A_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&dB, B_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&dC_offsets, (params.m + 1) * sizeof(int64_t)));
  CHECK_CUDA(cudaMalloc((void**)&dC_columns, c_true_nnz * sizeof(int64_t)));
  CHECK_CUDA(cudaMalloc((void**)&dC_values, c_true_nnz * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(dA, hA, A_size * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB, B_size * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
    dC_offsets, hC_offsets.data(), (params.m + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
    dC_columns, hC_columns.data(), c_true_nnz * sizeof(int64_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(
    cudaMemcpy(dC_values, hC_values.data(), c_true_nnz * sizeof(float), cudaMemcpyHostToDevice));
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
    cusparseCreateDnMat(&matA, params.m, params.k, lda, dA, CUDA_R_32F, CUSPARSE_ORDER_ROW))
  // Create dense matrix B
  CHECK_CUSPARSE(
    cusparseCreateDnMat(&matB, params.k, params.n, ldb, dB, CUDA_R_32F, CUSPARSE_ORDER_COL))
  // Create sparse matrix C in CSR format
  CHECK_CUSPARSE(cusparseCreateCsr(&matC,
                                   params.m,
                                   params.n,
                                   c_true_nnz,
                                   dC_offsets,
                                   dC_columns,
                                   dC_values,
                                   CUSPARSE_INDEX_64I,
                                   CUSPARSE_INDEX_64I,
                                   CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_32F))
  // execute SpMM
  cudaStream_t stream;

  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUSPARSE(cusparseSetStream(handle, stream));
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
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize * 4))

  // execute preprocess (optional)
  //   CHECK_CUSPARSE(cusparseSDDMM_preprocess(handle,
  //                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
  //                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
  //                                           &params.alpha,
  //                                           matA,
  //                                           matB,
  //                                           &params.beta,
  //                                           matC,
  //                                           CUDA_R_32F,
  //                                           CUSPARSE_SDDMM_ALG_DEFAULT,
  //                                           dBuffer))

  timer.start();
  CHECK_CUSPARSE(cusparseSDDMM(handle,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &params.alpha,
                               matA,
                               matB,
                               &params.beta,
                               matC,
                               CUDA_R_32F,
                               CUSPARSE_SDDMM_ALG_DEFAULT,
                               dBuffer))

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
}

int main(void)
{
//   std::vector<SDDMMBenchParams> cases{{1024 * 1024, 128, 1024, 0.01, 1.0f, 0.0f}};
//
//   auto timer             = Timer<double>();
//   int times              = 3;
//   double accumulated_dur = 0.0;
//   for (auto params : cases) {
//     test_main(params, timer);
//     for (int time = 0; time < times; time++) {
//       test_main(params, timer);
//       accumulated_dur += timer.getResult();
//     }
//     std::cout << accumulated_dur / static_cast<double>(1.0 * times) << std::endl;
//   }

  std::vector<bool> c_dense_data_h { true, true, true,
                                     false, true, false,
                                     true, true, true,
                                     true, false, true};

  size_t c_true_nnz = 9;

  std::cout << "c_true_nnz: " << c_true_nnz << std::endl;

  std::vector<float> hC_values(c_true_nnz);
  std::vector<int64_t> hC_columns(c_true_nnz);
  std::vector<int64_t> hC_offsets(4 + 1);

  convert_to_csr_problem<float, int64_t>(
    c_dense_data_h, 4, 3, hC_values.data(), hC_columns.data(), hC_offsets.data());
  for(auto a: hC_values) std::cout << a << ", ";
  std::cout << std::endl;
  for(auto a: hC_columns) std::cout << a << ", ";
  std::cout << std::endl;
  for(auto a: hC_offsets) std::cout << a << ", ";
  std::cout << std::endl;

  return EXIT_SUCCESS;
}