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

void test_main(BenchParams& params, Timer<double>& timer, size_t& bufferSize)
{
  // Host problem definition
  size_t lda    = params.k;
  size_t ldb    = params.k;
  size_t A_size = params.m * params.k;
  size_t B_size = params.k * params.n;
  size_t C_size = params.m * params.n;

  auto opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

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
      cusparseCreateDnMat(&matA, params.k, params.m, lda, dA, CUDA_R_32F, CUSPARSE_ORDER_COL))
  }

  if (!params.b_is_row) {
    CHECK_CUSPARSE(
      cusparseCreateDnMat(&matB, params.k, params.n, ldb, dB, CUDA_R_32F, CUSPARSE_ORDER_COL))
  } else {
    CHECK_CUSPARSE(
      cusparseCreateDnMat(&matB, params.n, params.k, ldb, dB, CUDA_R_32F, CUSPARSE_ORDER_ROW))
  }

  // Perpare C and test
  for (float sp : {0.01, 0.1, 0, 2, 0.5}) {
    std::vector<bool> c_dense_data_h(C_size);
    size_t c_true_nnz = create_sparse_matrix(params.m, params.n, params.sparsity, c_dense_data_h);

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
                                            &bufferSize))
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

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

    std::cout << size_t(bufferSize / (1024 * 1024)) << "MB\t";
    std::cout << params.m << "\t" << params.k << "\t" << params.n << "\t" << params.sparsity
              << "\t\t" << params.alpha << "\t" << params.beta << "\t"
              << (params.a_is_row ? "row" : "col") << "\t" << (params.b_is_row ? "row" : "col")
              << "\t" << fixed << setprecision(3) << setw(6) << setfill(' ')
              << static_cast<float>(timer.getResult()) << "ms" << std::endl;

    CHECK_CUDA(cudaFree(dBuffer))
    CHECK_CUDA(cudaFree(dC_offsets))
    CHECK_CUDA(cudaFree(dC_columns))
    CHECK_CUDA(cudaFree(dC_values))
  }
  CHECK_CUDA(cudaStreamDestroy(stream));

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
    {1024 * 1024, 128, 1024, 0.01, 1.0f, 0.0f, true, false},
    {1024 * 1024, 1024, 1024, 0.01, 1.0f, 0.0f, true, false},
    {1024 * 1024, 1024, 10 * 1024, 0.01, 1.0f, 0.0f, true, false},
    {1024 * 1024 * 1024, 1024, 10 * 1024, 0.01, 1.0f, 0.0f, true, false}};

  auto timer             = Timer<double>();
  int times              = 2;
  size_t bufferSize      = 0;
  double accumulated_dur = 0.0;
  std::cout << "buffer\t"
            << "m\t"
            << "k\t"
            << "n\t"
            << "sparsity\t"
            << "alpha\t"
            << "beta\t"
            << "orderA\t"
            << "orderB\t"
            << "duration" << std::endl;
  std::cout
    << "----------------------------------------------------------------------------------------"
    << std::endl;
  for (auto params : cases) {
    bufferSize = 0;
    test_main(params, timer, bufferSize);  // warmup
    for (int time = 0; time < times; time++) {
      test_main(params, timer, bufferSize);
      accumulated_dur += timer.getResult();
    }
  }

  //   std::vector<bool> c_dense_data_h{
  //     true, true, true, false, true, false, true, true, true, true, false, true};
  //
  //   size_t c_true_nnz = 9;
  //
  //   std::cout << "c_true_nnz: " << c_true_nnz << std::endl;
  //
  //   std::vector<float> hC_values(c_true_nnz);
  //   std::vector<int64_t> hC_columns(c_true_nnz);
  //   std::vector<int64_t> hC_offsets(4 + 1);
  //
  //   convert_to_csr(c_dense_data_h, 4, 3, hC_values, hC_columns, hC_offsets);
  //   for (auto a : hC_values)
  //     std::cout << a << ", ";
  //   std::cout << std::endl;
  //   for (auto a : hC_columns)
  //     std::cout << a << ", ";
  //   std::cout << std::endl;
  //   for (auto a : hC_offsets)
  //     std::cout << a << ", ";
  //   std::cout << std::endl;

  return EXIT_SUCCESS;
}