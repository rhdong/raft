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

#include <raft/sparse/linalg/detail/sddmm.cuh>

namespace raft {
namespace sparse {
namespace linalg {

/**
 * @brief SDDMM function designed for handling all DENSE * DENSE
 * combinations of operand layouts for cuSparse.
 * It computes the following equation: Z = alpha . X * Y + beta . Z
 * where X,Y are device matrix views and Z is a CSR device matrix view
 * @tparam ValueType Data type of input/output matrices (float/double)
 * @tparam IndexType Type of Z
 * @tparam LayoutPolicyY layout of X
 * @tparam LayoutPolicyZ layout of Y
 * @tparam NZType Type of Z
 * @param[in] handle raft handle
 * @param[in] trans_x transpose operation for X
 * @param[in] trans_y transpose operation for Y
 * @param[in] alpha scalar
 * @param[in] x input raft::device_matrix_view
 * @param[in] y input raft::device_matrix_view
 * @param[in] beta scalar
 * @param[out] z output raft::device_csr_matrix_view
 */
template <typename ValueType,
          typename IndexType,
          typename LayoutPolicyX,
          typename LayoutPolicyY,
          typename NZType>
void sddmm(raft::resources const& handle,
           const bool trans_x,
           const bool trans_y,
           const ValueType* alpha,
           raft::device_matrix_view<const ValueType, IndexType, LayoutPolicyX> x,
           raft::device_matrix_view<const ValueType, IndexType, LayoutPolicyY> y,
           const ValueType* beta,
           raft::device_csr_matrix_view<ValueType, int, int, NZType> z)
{
  bool is_row_major = detail::is_row_major(x, y);

  auto descr_x = detail::create_descriptor(x, is_row_major);
  auto descr_y = detail::create_descriptor(y, is_row_major);
  auto descr_z = detail::create_descriptor(z);

  detail::sddmm(handle, trans_x, trans_y, is_row_major, alpha, descr_x, descr_y, beta, descr_z);

  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnMat(descr_x));
  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnMat(descr_y));
  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroySpMat(descr_z));
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // end namespace linalg
}  // end namespace sparse
}  // end namespace raft

#endif
