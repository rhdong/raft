/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "../../test_utils.cuh"

#include <raft/core/bitmap.cuh>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/neighbors/brute_force.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/cuda_utils.cuh>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <queue>
#include <random>
#include <unordered_set>
#include <vector>

namespace raft::neighbors::brute_force {

template <typename index_t>
struct PrefilteredBruteForceInputs {
  index_t n_rows;
  index_t n_cols;
  index_t dim;
  index_t top_k;
  float sparsity;
  raft::distance::DistanceType metric = raft::distance::DistanceType::L2SqrtUnexpanded;
  bool select_min                     = true;
};

template <typename T>
struct CompareApproxWithInf {
  CompareApproxWithInf(T eps_) : eps(eps_) {}
  bool operator()(const T& a, const T& b) const
  {
    if (std::isinf(a) && std::isinf(b)) return true;
    T diff  = std::abs(a - b);
    T m     = std::max(std::abs(a), std::abs(b));
    T ratio = diff > eps ? diff / m : diff;

    return (ratio <= eps);
  }

 private:
  T eps;
};

template <typename value_t, typename index_t, typename bitmap_t = uint32_t>
class PrefilteredBruteForceTest
  : public ::testing::TestWithParam<PrefilteredBruteForceInputs<index_t>> {
 public:
  PrefilteredBruteForceTest()
    : stream(resource::get_cuda_stream(handle)),
      params(::testing::TestWithParam<PrefilteredBruteForceInputs<index_t>>::GetParam()),
      filter_d(0, stream),
      dataset_d(0, stream),
      queries_d(0, stream),
      out_val_d(0, stream),
      out_val_expected_d(0, stream),
      out_idx_d(0, stream),
      out_idx_expected_d(0, stream)
  {
  }

 protected:
  index_t create_sparse_matrix(index_t m, index_t n, float sparsity, std::vector<bitmap_t>& bitmap)
  {
    index_t total    = static_cast<index_t>(m * n);
    index_t num_ones = static_cast<index_t>((total * 1.0f) * sparsity);
    index_t res      = num_ones;

    for (auto& item : bitmap) {
      item = static_cast<bitmap_t>(0);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<index_t> dis(0, total - 1);

    while (num_ones > 0) {
      index_t index = dis(gen);

      bitmap_t& element    = bitmap[index / (8 * sizeof(bitmap_t))];
      index_t bit_position = index % (8 * sizeof(bitmap_t));

      if (((element >> bit_position) & 1) == 0) {
        element |= (static_cast<index_t>(1) << bit_position);
        num_ones--;
      }
    }
    return res;
  }

  void cpu_convert_to_csr(std::vector<bitmap_t>& bitmap,
                          index_t rows,
                          index_t cols,
                          std::vector<index_t>& indices,
                          std::vector<index_t>& indptr)
  {
    index_t offset_indptr   = 0;
    index_t offset_values   = 0;
    indptr[offset_indptr++] = 0;

    index_t index        = 0;
    bitmap_t element     = 0;
    index_t bit_position = 0;

    for (index_t i = 0; i < rows; ++i) {
      for (index_t j = 0; j < cols; ++j) {
        index        = i * cols + j;
        element      = bitmap[index / (8 * sizeof(bitmap_t))];
        bit_position = index % (8 * sizeof(bitmap_t));

        if (((element >> bit_position) & 1)) {
          indices[offset_values] = static_cast<index_t>(j);
          offset_values++;
        }
      }
      indptr[offset_indptr++] = static_cast<index_t>(offset_values);
    }
  }

  void cpu_sddmm(const std::vector<value_t>& A,
                 const std::vector<value_t>& B,
                 std::vector<value_t>& vals,
                 const std::vector<index_t>& cols,
                 const std::vector<index_t>& row_ptrs,
                 bool is_row_major_A,
                 bool is_row_major_B,
                 value_t alpha = 1.0,
                 value_t beta  = 0.0)
  {
    if (params.n_rows * params.top_k != static_cast<index_t>(A.size()) ||
        params.top_k * params.n_cols != static_cast<index_t>(B.size())) {
      std::cerr << "Matrix dimensions and vector size do not match!" << std::endl;
      return;
    }

    bool trans_a = is_row_major_A;
    bool trans_b = is_row_major_B;

    for (index_t i = 0; i < params.n_rows; ++i) {
      for (index_t j = row_ptrs[i]; j < row_ptrs[i + 1]; ++j) {
        value_t sum = 0;
        for (index_t l = 0; l < params.top_k; ++l) {
          index_t a_index = trans_a ? i * params.top_k + l : l * params.n_rows + i;
          index_t b_index = trans_b ? l * params.n_cols + cols[j] : cols[j] * params.top_k + l;
          sum += A[a_index] * B[b_index];
        }
        vals[j] = alpha * sum + beta * vals[j];
      }
    }
  }

  void cpu_select_k(const std::vector<index_t>& indptr_h,
                    const std::vector<index_t>& indices_h,
                    const std::vector<value_t>& values_h,
                    std::optional<std::vector<index_t>>& in_idx_h,
                    index_t n_rows,
                    index_t n_cols,
                    index_t top_k,
                    std::vector<value_t>& out_values_h,
                    std::vector<index_t>& out_indices_h,
                    bool select_min = true)
  {
    auto comp = [select_min](const std::pair<value_t, index_t>& a,
                             const std::pair<value_t, index_t>& b) {
      return select_min ? a.first < b.first : a.first >= b.first;
    };

    for (index_t row = 0; row < n_rows; ++row) {
      std::priority_queue<std::pair<value_t, index_t>,
                          std::vector<std::pair<value_t, index_t>>,
                          decltype(comp)>
        pq(comp);

      for (index_t idx = indptr_h[row]; idx < indptr_h[row + 1]; ++idx) {
        pq.push({values_h[idx], (in_idx_h.has_value()) ? (*in_idx_h)[idx] : indices_h[idx]});
        if (pq.size() > size_t(top_k)) { pq.pop(); }
      }

      std::vector<std::pair<value_t, index_t>> row_pairs;
      while (!pq.empty()) {
        row_pairs.push_back(pq.top());
        pq.pop();
      }

      if (select_min) {
        std::sort(row_pairs.begin(), row_pairs.end(), [](const auto& a, const auto& b) {
          return a.first <= b.first;
        });
      } else {
        std::sort(row_pairs.begin(), row_pairs.end(), [](const auto& a, const auto& b) {
          return a.first >= b.first;
        });
      }
      for (index_t col = 0; col < top_k; col++) {
        if (col < index_t(row_pairs.size())) {
          out_values_h[row * top_k + col]  = row_pairs[col].first;
          out_indices_h[row * top_k + col] = row_pairs[col].second;
        }
      }
    }
  }

  void random_array(value_t* array, size_t size)
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<value_t> dis(-10.0, 10.0);
    std::unordered_set<value_t> uset;

    while (uset.size() < size) {
      uset.insert(dis(gen));
    }
    typename std::unordered_set<value_t>::iterator it = uset.begin();
    for (size_t i = 0; i < size; ++i) {
      array[i] = *(it++);
    }
  }

  void SetUp() override
  {
    index_t element = raft::ceildiv(params.n_rows * params.n_cols, index_t(sizeof(bitmap_t) * 8));
    std::vector<bitmap_t> filter_h(element);

    nnz = create_sparse_matrix(params.n_rows, params.n_cols, params.sparsity, filter_h);

    index_t dataset_size = params.n_rows * params.dim;
    index_t queries_size = params.dim * params.n_cols;

    std::vector<value_t> dataset_h(dataset_size);
    std::vector<value_t> queries_h(queries_size);

    dataset_d.resize(dataset_size, stream);
    queries_d.resize(queries_size, stream);

    auto blobs_in_val =
      raft::make_device_matrix<value_t, index_t>(handle, 1, dataset_size + queries_size);
    auto labels = raft::make_device_vector<index_t, index_t>(handle, 1);

    raft::random::make_blobs<value_t, index_t>(blobs_in_val.data_handle(),
                                               labels.data_handle(),
                                               1,
                                               dataset_size + queries_size,
                                               1,
                                               stream,
                                               false,
                                               nullptr,
                                               nullptr,
                                               value_t(1.0),
                                               false,
                                               value_t(-1.0f),
                                               value_t(1.0f),
                                               uint64_t(2024));

    raft::copy(dataset_h.data(), blobs_in_val.data_handle(), dataset_size, stream);
    raft::copy(dataset_d.data(), blobs_in_val.data_handle(), dataset_size, stream);

    raft::copy(queries_h.data(), blobs_in_val.data_handle() + dataset_size, queries_size, stream);
    raft::copy(queries_d.data(), blobs_in_val.data_handle() + dataset_size, queries_size, stream);

    resource::sync_stream(handle);

    std::vector<value_t> values_h(nnz);
    std::vector<index_t> indices_h(nnz);
    std::vector<index_t> indptr_h(params.n_rows + 1);

    filter_d.resize(filter_h.size(), stream);
    cpu_convert_to_csr(filter_h, params.n_rows, params.n_cols, indices_h, indptr_h);

    cpu_sddmm(dataset_h, queries_h, values_h, indices_h, indptr_h, true, true);

    std::vector<value_t> out_val_h(params.n_rows * params.top_k,
                                   std::numeric_limits<value_t>::infinity());
    std::vector<index_t> out_idx_h(params.n_rows * params.top_k, static_cast<index_t>(0));

    out_val_d.resize(params.n_rows * params.top_k, stream);
    out_idx_d.resize(params.n_rows * params.top_k, stream);

    update_device(out_val_d.data(), out_val_h.data(), out_val_h.size(), stream);
    update_device(out_idx_d.data(), out_idx_d.data(), out_idx_d.size(), stream);
    update_device(filter_d.data(), filter_h.data(), filter_h.size(), stream);

    resource::sync_stream(handle);

    std::optional<std::vector<index_t>> optional_indices_h = std::nullopt;

    cpu_select_k(indptr_h,
                 indices_h,
                 values_h,
                 optional_indices_h,
                 params.n_rows,
                 params.n_cols,
                 params.top_k,
                 out_val_h,
                 out_idx_h,
                 params.select_min);

    out_val_expected_d.resize(params.n_rows * params.top_k, stream);
    out_idx_expected_d.resize(params.n_rows * params.top_k, stream);

    update_device(out_val_expected_d.data(), out_val_h.data(), out_val_h.size(), stream);
    update_device(out_idx_expected_d.data(), out_idx_d.data(), out_idx_d.size(), stream);

    resource::sync_stream(handle);
  }

  void Run()
  {
    auto dataset_raw = raft::make_device_matrix_view<const value_t, index_t, raft::row_major>(
      (const value_t*)dataset_d.data(), params.n_rows, params.dim);

    auto queries = raft::make_device_matrix_view<const value_t, index_t>(
      (const value_t*)queries_d.data(), params.dim, params.n_cols);

    brute_force::index_params index_params{};
    index_params.metric     = params.metric;
    index_params.metric_arg = 0;

    auto dataset = brute_force::build(handle, index_params, dataset_raw);

    auto filter = raft::core::bitmap_view(
      (const bitmap_t*)filter_d.data(), params.n_rows, params.n_cols);

    auto out_val = raft::make_device_matrix_view<value_t, index_t, raft::row_major>(
      out_val_d.data(), params.n_rows, params.top_k);
    auto out_idx = raft::make_device_matrix_view<index_t, index_t, raft::row_major>(
      out_idx_d.data(), params.n_rows, params.top_k);

    brute_force::search_with_filtering(handle, dataset, queries, filter, out_val, out_idx);

    ASSERT_TRUE(raft::devArrMatch<index_t>(out_idx_expected_d.data(),
                                           out_idx.data_handle(),
                                           params.n_rows * params.top_k,
                                           raft::Compare<index_t>(),
                                           stream));

    ASSERT_TRUE(raft::devArrMatch<value_t>(out_val_expected_d.data(),
                                           out_val.data_handle(),
                                           params.n_rows * params.top_k,
                                           CompareApproxWithInf<value_t>(1e-6f),
                                           stream));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  PrefilteredBruteForceInputs<index_t> params;

  index_t nnz;

  rmm::device_uvector<value_t> dataset_d;
  rmm::device_uvector<value_t> queries_d;
  rmm::device_uvector<bitmap_t> filter_d;

  rmm::device_uvector<value_t> out_val_d;
  rmm::device_uvector<value_t> out_val_expected_d;

  rmm::device_uvector<index_t> out_idx_d;
  rmm::device_uvector<index_t> out_idx_expected_d;
};

using PrefilteredBruteForceTest_double_int64 = PrefilteredBruteForceTest<float, int64_t>;
TEST_P(PrefilteredBruteForceTest_double_int64, Result) { Run(); }

template <typename index_t>
const std::vector<PrefilteredBruteForceInputs<index_t>> selectk_inputs = {{10, 32, 20, 10, 0.0},
                                                                          {10, 32, 20, 10, 0.0},
                                                                          {10, 32, 20, 10, 0.01},
                                                                          {10, 32, 20, 10, 0.1},
                                                                          {10, 32, 500, 251, 0.1},
                                                                          {10, 32, 500, 251, 0.6}};

INSTANTIATE_TEST_CASE_P(PrefilteredBruteForceTest,
                        PrefilteredBruteForceTest_double_int64,
                        ::testing::ValuesIn(selectk_inputs<int64_t>));

}  // namespace raft::neighbors::brute_force
