/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <common/benchmark.hpp>
#include <cusparse_v2.h>
#include <memory>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/distance/distance.cuh>
#include <raft/distance/distance_types.hpp>
#include <raft/random/rng.cuh>
#include <raft/sparse/linalg/sddmm.cuh>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace raft::bench::linalg {

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

using row_major = row_major;
using col_major = col_major;

template <typename ValueType>
struct SDDMMBenchParams {
  size_t m;  // m parameter of the SDDMM
  size_t k;  // k parameter of the SDDMM
  size_t n;  // n parameter of the SDDMM
  float sparsity;
  ValueType alpha = 1.0;
  ValueType beta  = 0.0;
};

enum Alg { SDDMM, Inner };

template <typename ValueType,
          typename LayoutPolicyA = row_major,
          typename LayoutPolicyB = col_major,
          const int SDDMMorInner = Alg::SDDMM,
          typename IndexType     = int64_t>
struct SDDMMBench : public fixture {
  SDDMMBench(const SDDMMBenchParams<ValueType>& p)
    : params(p),
      handle(stream),
      a_data_d(0, stream),
      b_data_d(0, stream),
      c_indptr_d(0, stream),
      c_indices_d(0, stream),
      c_data_d(0, stream),
      c_dense_data_d(0, stream)
  {
    a_data_d.resize(params.m * params.k, stream);
    b_data_d.resize(params.k * params.n, stream);

    raft::random::RngState rng(123456ULL);
    raft::random::uniform(
      handle, rng, a_data_d.data(), params.m * params.k, ValueType(-1.0), ValueType(1.0));
    raft::random::uniform(
      handle, rng, b_data_d.data(), params.k * params.n, ValueType(-1.0), ValueType(1.0));

    // init C mask
    std::vector<bool> c_dense_data_h(params.m * params.n);

    c_true_nnz = create_sparse_matrix(params.m, params.n, params.sparsity, c_dense_data_h);
    std::vector<ValueType> values(c_true_nnz);
    std::vector<IndexType> indices(c_true_nnz);
    std::vector<IndexType> indptr(params.m + 1);

    c_data_d.resize(c_true_nnz, stream);
    c_indptr_d.resize(params.m + 1, stream);
    c_indices_d.resize(c_true_nnz, stream);

    if (SDDMMorInner == Alg::Inner) { c_dense_data_d.resize(params.m * params.n, stream); }

    convert_to_csr(c_dense_data_h, params.m, params.n, values, indices, indptr);
    RAFT_EXPECTS(c_true_nnz == c_indices_d.size(),
                 "Something wrong. The c_true_nnz != c_indices_d.size()!");

    update_device(c_data_d.data(), values.data(), c_true_nnz, stream);
    update_device(c_indices_d.data(), indices.data(), c_true_nnz, stream);
    update_device(c_indptr_d.data(), indptr.data(), params.m + 1, stream);
  }

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
  //   void convert_to_csr(std::vector<bool>& matrix,
  //                       size_t rows,
  //                       size_t cols,
  //                       float* values,
  //                       IndexType* indices,
  //                       IndexType* indptr)
  //   {
  //     IndexType offset_indptr = 0;
  //     IndexType offset_values = 0;
  //     indptr[offset_indptr++] = 0;
  //
  //     for (size_t i = 0; i < rows; ++i) {
  //       for (size_t j = 0; j < cols; ++j) {
  //         if (matrix[i * cols + j]) {
  //           values[offset_values]  = static_cast<float>(1.0f);
  //           indices[offset_values] = static_cast<IndexType>(j);
  //           offset_values++;
  //         }
  //       }
  //       indptr[offset_indptr++] = static_cast<IndexType>(offset_values);
  //     }
  //   }
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

  ~SDDMMBench() {}

  void test_main()
  {
    // Host problem definition
    size_t lda = params.k;
    size_t ldb = params.k;
    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseDnMatDescr_t matA, matB;
    cusparseSpMatDescr_t matC;
    void* dBuffer     = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle))
    // Create dense matrix A
    CHECK_CUSPARSE(cusparseCreateDnMat(
      &matA, params.m, params.k, lda, a_data_d.data(), CUDA_R_32F, CUSPARSE_ORDER_ROW))
    // Create dense matrix B
    CHECK_CUSPARSE(cusparseCreateDnMat(
      &matB, params.k, params.n, ldb, b_data_d.data(), CUDA_R_32F, CUSPARSE_ORDER_COL))
    // Create sparse matrix C in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&matC,
                                     params.m,
                                     params.n,
                                     c_true_nnz,
                                     c_indptr_d.data(),
                                     c_indices_d.data(),
                                     c_data_d.data(),
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

    //     timer.start();
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
    //     timer.end();
    CHECK_CUDA(cudaStreamDestroy(stream));

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroyDnMat(matA))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
    CHECK_CUSPARSE(cusparseDestroySpMat(matC))
    CHECK_CUSPARSE(cusparseDestroy(handle))

    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA(cudaFree(dBuffer))
  }

  void run_benchmark(::benchmark::State& state) override
  {
    // make matrix view
    auto a = raft::make_device_matrix_view<const ValueType, IndexType, LayoutPolicyA>(
      a_data_d.data(), params.m, params.k);

    auto b = raft::make_device_matrix_view<const ValueType, IndexType, LayoutPolicyB>(
      b_data_d.data(), params.k, params.n);

    auto c_structure = raft::make_device_compressed_structure_view<int64_t, int64_t, int64_t>(
      c_indptr_d.data(),
      c_indices_d.data(),
      params.m,
      params.n,
      static_cast<IndexType>(c_indices_d.size()));

    auto c = raft::make_device_csr_matrix_view<ValueType>(c_data_d.data(), c_structure);
    raft::resource::get_cusparse_handle(handle);
    auto old_mr = rmm::mr::get_current_device_resource();
    if (SDDMMorInner == Alg::SDDMM) {
      rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
        old_mr, 10 * 1024 * 1024 * 1024ull);
      rmm::mr::set_current_device_resource(&pool_mr);
    }

    RAFT_CUDA_TRY(cudaStreamSynchronize(resource::get_cuda_stream(handle)));

    raft::sparse::linalg::sddmm(handle,
                                a,
                                b,
                                c,
                                raft::make_host_scalar_view<ValueType>(&params.alpha),
                                raft::make_host_scalar_view<ValueType>(&params.beta));

    loop_on_state(state, [this, &a, &b, &c]() {
      if (SDDMMorInner == Alg::SDDMM) {
        raft::sparse::linalg::sddmm(handle,
                                    a,
                                    b,
                                    c,
                                    raft::make_host_scalar_view<ValueType>(&params.alpha),
                                    raft::make_host_scalar_view<ValueType>(&params.beta));
        RAFT_CUDA_TRY(cudaStreamSynchronize(resource::get_cuda_stream(handle)));
        //         test_main();
      } else {
        raft::distance::pairwise_distance(handle,
                                          a_data_d.data(),
                                          b_data_d.data(),
                                          c_dense_data_d.data(),
                                          static_cast<int>(params.m),
                                          static_cast<int>(params.n),
                                          static_cast<int>(params.k),
                                          raft::distance::DistanceType::InnerProduct,
                                          std::is_same_v<LayoutPolicyA, row_major>);
        RAFT_CUDA_TRY(cudaStreamSynchronize(resource::get_cuda_stream(handle)));
      }
    });

    rmm::mr::set_current_device_resource(old_mr);
  }

 private:
  const raft::device_resources handle;
  SDDMMBenchParams<ValueType> params;

  rmm::device_uvector<ValueType> a_data_d;
  rmm::device_uvector<ValueType> b_data_d;
  rmm::device_uvector<ValueType> c_dense_data_d;

  size_t c_true_nnz = 0;
  rmm::device_uvector<IndexType> c_indptr_d;
  rmm::device_uvector<IndexType> c_indices_d;
  rmm::device_uvector<ValueType> c_data_d;
};

template <typename ValueType>
static std::vector<SDDMMBenchParams<ValueType>> getInputs()
{
  std::vector<SDDMMBenchParams<ValueType>> param_vec;
  struct TestSize {
    size_t m;
    size_t k;
    size_t n;
    float sparsity;
  };

std::vector<TestSize> data_size{
                               {1024 * 1024, 1024, 2 * 1024, 1.0f},
                               {1024 * 1024, 128, 1024, 1.0f},
                               {1024 * 1024, 1024, 1024, 1.0f},
                               {1024 * 1024, 1024, 2 * 1024, 1.0f},
                               {1024 * 1024, 128, 1024, 1.0f},
                               {1024 * 1024, 1024, 1024, 1.0f},
                               {1024 * 1024, 1024, 2 * 1024, 1.0f},
                               {1024 * 1024, 128, 1024, 1.0f},
                               {1024 * 1024, 1024, 1024, 1.0f},
                               {1024 * 1024, 1024, 2 * 1024, 1.0f}
  };

  param_vec.reserve(data_size.size());
  for (TestSize s : data_size) {
    param_vec.push_back(SDDMMBenchParams<ValueType>({s.m, s.k, s.n, s.sparsity}));
  }
  return param_vec;
}

// RAFT_BENCH_REGISTER((SDDMMBench<float, row_major, col_major, Alg::SDDMM>), "",
// getInputs<float>()); RAFT_BENCH_REGISTER((SDDMMBench<float, col_major, row_major, Alg::SDDMM>),
// "", getInputs<float>()); RAFT_BENCH_REGISTER((SDDMMBench<float, row_major, row_major,
// Alg::SDDMM>),
// "", getInputs<float>()); RAFT_BENCH_REGISTER((SDDMMBench<float, col_major, col_major,
// Alg::SDDMM>), "", getInputs<float>());
//
RAFT_BENCH_REGISTER((SDDMMBench<float, row_major, col_major, Alg::Inner>), "", getInputs<float>());

}  // namespace raft::bench::linalg
