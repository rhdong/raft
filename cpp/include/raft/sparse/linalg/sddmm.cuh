/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <raft/sparse/linalg/detail/sddmm.hpp>
#include <raft/sparse/linalg/detail/utils.cuh>
#include <raft/util/input_validation.hpp>

namespace raft {
namespace sparse {
namespace linalg {

/**
 * @brief This function performs the multiplication of dense matrix A and dense matrix B,
 * followed by an element-wise multiplication with the sparsity pattern of C.
 * It computes the following equation: C = alpha · (A * B ∘ spy(C)) + beta · C
 * where A,B are device matrix views and C is a CSR device matrix view
 *
 * @note  For best performance, the order of A and B should be different.
 *        If the order of A and B are the same, B will be transposed.
 *
 * @tparam ValueType Data type of input/output matrices (float/double)
 * @tparam IndexType Type of C
 * @tparam LayoutPolicyA layout of A
 * @tparam LayoutPolicyB layout of B
 * @tparam NZType Type of C
 *
 * @param[in] handle raft handle
 * @param[in] a input raft::device_matrix_view
 * @param[in] b input raft::device_matrix_view
 * @param[in/out] c output raft::device_csr_matrix_view
 * @param[in] alpha input raft::host_scalar_view
 * @param[in] beta input raft::host_scalar_view
 */
template <typename ValueType,
          typename IndexType,
          typename NZType,
          typename LayoutPolicyA,
          typename LayoutPolicyB>
void sddmm(raft::resources const& handle,
           raft::device_matrix_view<const ValueType, IndexType, LayoutPolicyA> a,
           raft::device_matrix_view<const ValueType, IndexType, LayoutPolicyB> b,
           raft::device_csr_matrix_view<ValueType, int64_t, int64_t, NZType> c,
           raft::host_scalar_view<ValueType> alpha,
           raft::host_scalar_view<ValueType> beta)
{
  RAFT_EXPECTS(raft::is_row_or_column_major(a), "A is not contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(b), "B is not contiguous");

  static_assert(std::is_same_v<ValueType, float> || std::is_same_v<ValueType, double>,
                "The `ValueType` of sddmm only supports float/double.");

  auto descr_a = detail::create_descriptor(a, false);
  auto descr_b = detail::create_descriptor(b, false);
  auto descr_c = detail::create_descriptor(c, false);

  detail::sddmm(handle,
                descr_a,
                descr_b,
                descr_c,
                raft::is_row_major(a),
                raft::is_row_major(b),
                alpha.data_handle(),
                beta.data_handle());

  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnMat(descr_a));
  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnMat(descr_b));
  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroySpMat(descr_c));
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // end namespace linalg
}  // end namespace sparse
}  // end namespace raft
