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

#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/util/input_validation.hpp>

namespace raft {
namespace sparse {
namespace linalg {
namespace detail {

/**
 * @brief create a cuSparse dense descriptor
 * @tparam ValueType Data type of dense_view (float/double)
 * @tparam IndexType Type of dense_view
 * @tparam LayoutPolicy layout of dense_view
 * @param[in] dense_view input raft::device_matrix_view
 * @returns dense matrix descriptor to be used by cuSparse API
 */
template <typename ValueType, typename IndexType, typename LayoutPolicy>
cusparseDnMatDescr_t create_descriptor(
  raft::device_matrix_view<ValueType, IndexType, LayoutPolicy>& dense_view,
  bool transpose_view = false)
{
  cusparseDnMatDescr_t descr;
  bool is_row_major = raft::is_row_major(dense_view);
  if (!transpose_view) {
    auto order   = is_row_major ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;
    IndexType ld = is_row_major ? dense_view.stride(0) : dense_view.stride(1);
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednmat(
      &descr,
      dense_view.extent(0),
      dense_view.extent(1),
      ld,
      const_cast<std::remove_const_t<ValueType>*>(dense_view.data_handle()),
      order));
  } else {
    auto order   = is_row_major ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_RAW;
    IndexType ld = is_row_major ? dense_view.stride(1) : dense_view.stride(0);
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednmat(
      &descr,
      dense_view.extent(1),
      dense_view.extent(0),
      ld,
      const_cast<std::remove_const_t<ValueType>*>(dense_view.data_handle()),
      order));
  }
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
  raft::device_csr_matrix_view<ValueType, IndptrType, IndicesType, NZType>& sparse_view,
  bool transpose_view = false)
{
  cusparseSpMatDescr_t descr;
  auto csr_structure = sparse_view.structure_view();
  if (!transpose_view) {
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatecsr(
      &descr,
      static_cast<int64_t>(csr_structure.get_n_rows()),
      static_cast<int64_t>(csr_structure.get_n_cols()),
      static_cast<int64_t>(csr_structure.get_nnz()),
      const_cast<IndptrType*>(csr_structure.get_indptr().data()),
      const_cast<IndicesType*>(csr_structure.get_indices().data()),
      const_cast<std::remove_const_t<ValueType>*>(sparse_view.get_elements().data())));
  } else {
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatecsr(
      &descr,
      static_cast<int64_t>(csr_structure.get_n_cols()),
      static_cast<int64_t>(csr_structure.get_n_rows()),
      static_cast<int64_t>(csr_structure.get_nnz()),
      const_cast<IndptrType*>(csr_structure.get_indptr().data()),
      const_cast<IndicesType*>(csr_structure.get_indices().data()),
      const_cast<std::remove_const_t<ValueType>*>(sparse_view.get_elements().data())));
  }
  return descr;
}

}  // end namespace detail
}  // end namespace linalg
}  // end namespace sparse
}  // end namespace raft
