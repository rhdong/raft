/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#pragma once

#include <raft/core/bitmap.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/handle.hpp>
#include <raft/linalg/matrix_mul.hpp>

#include <raft/core/resource/cusparse_handle.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>

namespace raft {
namespace linalg {
namespace detail {

template <typename value_type, typename index_type = int64_t>
void masked_matmul(raft::resources const& res,
                   const raft::device_matrix_view<const value_type, index_type, row_major>& A,
                   const raft::device_matrix_view<const value_type, index_type, row_major>& B,
                   raft::device_matrix_view<value_type, index_type, row_major>& C,
                   const raft::core::bitset& mask)
{
    std::vector<value_idx> indptr_h  = params.indptr_h;
    std::vector<value_idx> indices_h = params.indices_h;
    std::vector<value_t> data_h      = params.data_h;

    auto stream = resource::get_cuda_stream(handle);
    indptr.resize(indptr_h.size(), stream);
    indices.resize(indices_h.size(), stream);
    data.resize(data_h.size(), stream);

    auto converted_C = raft::device_csr_matrix_view<value_type, index_type, row_major> bitset::to_csr();
    /*
    #include <thrust/scan.h>

    int data[6] = {1, 0, 2, 2, 1, 3};

    thrust::exclusive_scan(data, data + 6, data); // in-place scan
    */

    auto x_structure = raft::make_device_compressed_structure_view<value_idx, value_idx, value_idx>(
      indptr.data(),
      indices.data(),
      static_cast<value_idx>(params.indptr_h.size() - 1),
      params.n_cols,
      static_cast<value_idx>(params.indices_h.size()));
    auto x = raft::make_device_csr_matrix_view<const value_t>(data.data(), x_structure);

  raft::device_csr_matrix_view<ValueType, int, int, NZType> C =
    mask.to_csr(mask, raft::resource::get_cuda_stream(res));

  RAFT_CUSPARSE_TRY(raft::sparse::detail::sddmm(res,
                                                trans_x,
                                                trans_y,
                                                static_cast<const ValueType>(1.0),
                                                A,
                                                B,
                                                static_cast<const ValueType>(0.0),
                                                converted_C));

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace detail
}  // namespace linalg
}  // namespace raft