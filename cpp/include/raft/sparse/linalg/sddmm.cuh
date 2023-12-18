/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

namespace raft {
namespace sparse {
namespace linalg {

/**
 * @brief This function performs the multiplication of dense matrix A and dense matrix B,
 * followed by an element-wise multiplication with the sparsity pattern of C.
 * It computes the following equation: C = alpha · (A * B ∘ spy(C)) + beta · C
 * where A,B are device matrix views and C is a CSR device matrix view
 *
 * @tparam ValueType Data type of input/output matrices (float/double)
 * @tparam IndexType Type of C
 * @tparam LayoutPolicyA layout of A
 * @tparam LayoutPolicyB layout of B
 * @tparam NZType Type of Cz
 *
 * @param[in] handle raft handle
 * @param[in] trans_a transpose operation for A
 * @param[in] trans_b transpose operation for B
 * @param[in] alpha scalar
 * @param[in] a input raft::device_matrix_view
 * @param[in] b input raft::device_matrix_view
 * @param[in] beta scalar
 * @param[out] c output raft::device_csr_matrix_view
 */
template <typename ValueType,
          typename IndexType,
          typename NZType,
          typename LayoutPolicyA,
          typename LayoutPolicyB>
void sddmm(raft::resources const& handle,
           const bool trans_a,
           const bool trans_b,
           const ValueType* alpha,
           raft::device_matrix_view<const ValueType, IndexType, LayoutPolicyA> a,
           raft::device_matrix_view<const ValueType, IndexType, LayoutPolicyB> b,
           const ValueType* beta,
           raft::device_csr_matrix_view<ValueType, int, int, NZType> c)
{
  bool is_row_major = detail::is_row_major(a, b);

  auto descr_a = detail::create_descriptor(a, is_row_major);
  auto descr_b = detail::create_descriptor(b, is_row_major);
  auto descr_c = detail::create_descriptor(c);

  detail::sddmm(handle, trans_a, trans_b, is_row_major, alpha, descr_a, descr_b, beta, descr_c);

  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnMat(descr_a));
  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnMat(descr_b));
  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroySpMat(descr_c));
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // end namespace linalg
}  // end namespace sparse
}  // end namespace raft
