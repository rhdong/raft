#include <chrono>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdio.h>
#include <stdlib.h>

using std::cout;
using std::endl;
using std::fixed;
using std::setfill;
using std::setprecision;
using std::setw;

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

struct BenchParams {
  size_t m;  // m parameter of the SDDMM
  size_t k;  // k parameter of the SDDMM
  size_t n;  // n parameter of the SDDMM
  float sparsity;
  float alpha = 1.0;
  float beta  = 0.0;
  bool a_is_row;
  bool b_is_row;
};

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

void convert_to_csr(std::vector<bool>& matrix,
                    size_t rows,
                    size_t cols,
                    std::vector<float>& values,
                    std::vector<int64_t>& indices,
                    std::vector<int64_t>& indptr)
{
  int64_t offset_indptr   = 0;
  int64_t offset_values   = 0;
  indptr[offset_indptr++] = 0;

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      if (matrix[i * cols + j]) {
        values[offset_values]  = static_cast<float>(1.0f);
        indices[offset_values] = static_cast<int64_t>(j);
        offset_values++;
      }
    }
    indptr[offset_indptr++] = static_cast<int64_t>(offset_values);
  }
}

void test_main(BenchParams& params, Timer<double>& timer)
{
  // Host problem definition
  size_t lda    = params.a_is_row ? params.k : params.m;
  size_t ldb    = params.b_is_row ? params.k : params.n;
  size_t A_size = params.m * params.k;
  size_t B_size = params.k * params.n;
  size_t C_size = params.m * params.n;

  auto opA = params.a_is_row ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
  auto opB = !params.b_is_row ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;

  std::vector<float> hA(A_size);
  std::vector<float> hB(B_size);

  uniform(hA.data(), A_size);
  uniform(hB.data(), B_size);

  //--------------------------------------------------------------------------
  // Device memory management
  int64_t *dC_offsets, *dC_columns;
  float *dC_values, *dB, *dA;
  CHECK_CUDA(cudaMalloc((void**)&dA, A_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void**)&dB, B_size * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dA, hA.data(), A_size * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), B_size * sizeof(float), cudaMemcpyHostToDevice));

  // Prepare A and B
  cusparseHandle_t handle = NULL;

  cusparseDnMatDescr_t matA, matB;
  cusparseSpMatDescr_t matC;
  void* dBuffer = NULL;
  CHECK_CUSPARSE(cusparseCreate(&handle))
  cudaStream_t stream;

  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUSPARSE(cusparseSetStream(handle, stream));

  // Create dense matrix A
  if (params.a_is_row) {
    CHECK_CUSPARSE(
      cusparseCreateDnMat(&matA, params.m, params.k, lda, dA, CUDA_R_32F, CUSPARSE_ORDER_ROW))
  } else {
    CHECK_CUSPARSE(
      cusparseCreateDnMat(&matA, params.m, params.k, lda, dA, CUDA_R_32F, CUSPARSE_ORDER_COL))
  }

  if (!params.b_is_row) {
    CHECK_CUSPARSE(
      cusparseCreateDnMat(&matB, params.k, params.n, ldb, dB, CUDA_R_32F, CUSPARSE_ORDER_COL))
  } else {
    CHECK_CUSPARSE(
      cusparseCreateDnMat(&matB, params.k, params.n, ldb, dB, CUDA_R_32F, CUSPARSE_ORDER_ROW))
  }

  // Perpare C and test
  // The first sparsity is only for warmup.
  std::vector<float> sparsity_list = {0.001, 0.01, 0.1, 0.19, 0.2, 0.5};
  size_t pre_buffer_size           = 0;
  bool warmup                      = true;
  int times                        = 3;
  for (float sp : sparsity_list) {
    std::vector<bool> c_dense_data_h(C_size);
    size_t c_true_nnz = create_sparse_matrix(params.m, params.n, sp, c_dense_data_h);

    std::vector<float> hC_values(c_true_nnz);
    std::vector<int64_t> hC_columns(c_true_nnz);
    std::vector<int64_t> hC_offsets(params.m + 1);

    convert_to_csr(c_dense_data_h, params.m, params.n, hC_values, hC_columns, hC_offsets);
    CHECK_CUDA(cudaMalloc((void**)&dC_offsets, (params.m + 1) * sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc((void**)&dC_columns, c_true_nnz * sizeof(int64_t)));
    CHECK_CUDA(cudaMalloc((void**)&dC_values, c_true_nnz * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(
      dC_offsets, hC_offsets.data(), (params.m + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
      dC_columns, hC_columns.data(), c_true_nnz * sizeof(int64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(
      cudaMemcpy(dC_values, hC_values.data(), c_true_nnz * sizeof(float), cudaMemcpyHostToDevice));
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
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
    // allocate an external buffer if needed
    size_t buffer_size = 0;
    CHECK_CUSPARSE(cusparseSDDMM_bufferSize(handle,
                                            opA,
                                            opB,
                                            &params.alpha,
                                            matA,
                                            matB,
                                            &params.beta,
                                            matC,
                                            CUDA_R_32F,
                                            CUSPARSE_SDDMM_ALG_DEFAULT,
                                            &buffer_size))
    CHECK_CUDA(cudaStreamSynchronize(stream));
    if (buffer_size > pre_buffer_size && dBuffer != NULL) {
      CHECK_CUDA(cudaFree(dBuffer))
      dBuffer = NULL;
    }
    if (dBuffer == NULL) {
      CHECK_CUDA(cudaMalloc(&dBuffer, buffer_size))
      pre_buffer_size = buffer_size;
    }

    // execute preprocess (optional)
    CHECK_CUSPARSE(cusparseSDDMM_preprocess(handle,
                                            opA,
                                            opB,
                                            &params.alpha,
                                            matA,
                                            matB,
                                            &params.beta,
                                            matC,
                                            CUDA_R_32F,
                                            CUSPARSE_SDDMM_ALG_DEFAULT,
                                            dBuffer))

    double accumulated_dur = 0.0;
    for (int time = 0; time < times; time++) {
      timer.start();
      CHECK_CUSPARSE(cusparseSDDMM(handle,
                                   opA,
                                   opB,
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
      accumulated_dur += timer.getResult();
    }

    CHECK_CUDA(cudaFree(dC_offsets))
    CHECK_CUDA(cudaFree(dC_columns))
    CHECK_CUDA(cudaFree(dC_values))

    if (!warmup) {
      std::cout << size_t(buffer_size / (1024 * 1024)) << "\t";
      std::cout << params.m << "\t\t" << params.k << "\t" << params.n << "\t" << sp << "\t\t"
                << fixed << setprecision(3) << setw(6) << setfill(' ') << params.alpha << "\t"
                << params.beta << "\t" << (params.a_is_row ? "row" : "col") << "\t"
                << (params.b_is_row ? "row" : "col") << "\t"
                << static_cast<float>(accumulated_dur / (times * 1.0f)) << "ms" << std::endl;
    }
    warmup = false;
  }
  CHECK_CUDA(cudaStreamDestroy(stream));
  if (dBuffer != NULL) { CHECK_CUDA(cudaFree(dBuffer)) }

  // destroy matrix/vector descriptors
  CHECK_CUSPARSE(cusparseDestroyDnMat(matA))
  CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
  CHECK_CUSPARSE(cusparseDestroySpMat(matC))
  CHECK_CUSPARSE(cusparseDestroy(handle))

  //--------------------------------------------------------------------------
  // device memory deallocation
  CHECK_CUDA(cudaFree(dA))
  CHECK_CUDA(cudaFree(dB))
}

int main(void)
{
  std::vector<BenchParams> cases{
//                                  {1024 * 1024, 128, 1024, 0.01, 1.0f, 0.0f, true, false},
//                                  {1024 * 1024, 1024, 1024, 0.01, 1.0f, 0.0f, true, false},
//                                  {1024 * 1024, 1024, 2 * 1024, 0.01, 1.0f, 0.0f, true, false},
//                                  {1024 * 1024, 128, 1024, 0.01, 1.0f, 0.0f, false, true},
//                                  {1024 * 1024, 1024, 1024, 0.01, 1.0f, 0.0f, false, true},
//                                  {1024 * 1024, 1024, 2 * 1024, 0.01, 1.0f, 0.0f, false, true},
                                 {1024 * 1024, 128, 1024, 0.01, 1.0f, 0.0f, true, true},
                                 {1024 * 1024, 1024, 1024, 0.01, 1.0f, 0.0f, true, true},
                                 {1024 * 1024, 1024, 2 * 1024, 0.01, 1.0f, 0.0f, true, true},
                                 {1024 * 1024, 128, 1024, 0.01, 1.0f, 0.0f, false, false},
                                 {1024 * 1024, 1024, 1024, 0.01, 1.0f, 0.0f, false, false},
                                 {1024 * 1024, 1024, 2 * 1024, 0.01, 1.0f, 0.0f, false, false}};

  auto timer = Timer<double>();
  std::cout << "-----------------------------------------------------------------------------------"
               "-------------"
            << std::endl;
  std::cout << "buffer\t"
            << "m\t\t"
            << "k\t"
            << "n\t"
            << "sparsity\t"
            << "alpha\t"
            << "beta\t"
            << "orderA\t"
            << "orderB\t"
            << "duration" << std::endl;
  std::cout << "-----------------------------------------------------------------------------------"
               "-------------"
            << std::endl;
  for (auto params : cases) {
    test_main(params, timer);
  }

  return EXIT_SUCCESS;
}