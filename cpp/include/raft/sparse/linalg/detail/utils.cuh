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

#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>

namespace raft {
namespace sparse {
namespace linalg {
namespace detail {

/**
 * @brief determine common data layout for both dense matrices
 * @tparam ValueType Data type of A, B (float/double)
 * @tparam IndexType Type of A, B
 * @tparam LayoutPolicyA layout of A
 * @tparam LayoutPolicyB layout of B
 * @param[in] a input raft::device_matrix_view
 * @param[in] b input raft::device_matrix_view
 * @returns dense matrix descriptor to be used by cuSparse API
 */
template <typename ValueTypeA,
          typename ValueTypeB,
          typename IndexType,
          typename LayoutPolicyA,
          typename LayoutPolicyB>
bool is_row_major(raft::device_matrix_view<ValueTypeA, IndexType, LayoutPolicyA>& a,
                  raft::device_matrix_view<ValueTypeB, IndexType, LayoutPolicyB>& b)
{
  bool is_row_major = a.stride(1) == 1 && b.stride(1) == 1;
  bool is_col_major = a.stride(0) == 1 && b.stride(0) == 1;
  ASSERT(is_row_major || is_col_major, "Both matrices need to be either row or col major");
  return is_row_major;
}

/**
 * @brief create a cuSparse dense descriptor
 * @tparam ValueType Data type of dense_view (float/double)
 * @tparam IndexType Type of dense_view
 * @tparam LayoutPolicy layout of dense_view
 * @param[in] dense_view input raft::device_matrix_view
 * @param[in] is_row_major data layout of raft::device_matrix_view
 * @returns dense matrix descriptor to be used by cuSparse API
 */
template <typename ValueType, typename IndexType, typename LayoutPolicy>
cusparseDnMatDescr_t create_descriptor(
  raft::device_matrix_view<ValueType, IndexType, LayoutPolicy>& dense_view, const bool is_row_major)
{
  auto order   = is_row_major ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;
  IndexType ld = is_row_major ? dense_view.stride(0) : dense_view.stride(1);
  cusparseDnMatDescr_t descr;
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednmat(
    &descr,
    dense_view.extent(0),
    dense_view.extent(1),
    ld,
    const_cast<std::remove_const_t<ValueType>*>(dense_view.data_handle()),
    order));
  return descr;
}

/**
 * @brief create a cuSparse sparse descriptor
 * @tparam ValueType Data type of sparse_view (float/double)
 * @tparam IndptrType Data type of csr_matrix_view index pointers
 * @tparam IndicesType Data type of csr_matrix_view indices
 * @tparam NZType Type of sparse_view
 * @param[in] sparse_view input raft::device_csr_matrix_view of size M rows x K columns
 * @returns sparse matrix descriptor to be used by cuSparse API
 */
template <typename ValueType, typename IndptrType, typename IndicesType, typename NZType>
cusparseSpMatDescr_t create_descriptor(
  raft::device_csr_matrix_view<ValueType, IndptrType, IndicesType, NZType>& sparse_view)
{
  cusparseSpMatDescr_t descr;
  auto csr_structure = sparse_view.structure_view();
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatecsr(
    &descr,
    static_cast<int64_t>(csr_structure.get_n_rows()),
    static_cast<int64_t>(csr_structure.get_n_cols()),
    static_cast<int64_t>(csr_structure.get_nnz()),
    const_cast<IndptrType*>(csr_structure.get_indptr().data()),
    const_cast<IndicesType*>(csr_structure.get_indices().data()),
    const_cast<std::remove_const_t<ValueType>*>(sparse_view.get_elements().data())));
  return descr;
}

}  // end namespace detail
}  // end namespace linalg
}  // end namespace sparse
}  // end namespace raft
