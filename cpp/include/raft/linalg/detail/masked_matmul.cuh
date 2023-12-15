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

template <typename T>
void masked_matmul(const raft::handle_t& handle,
                   const raft::device_matrix<T>& A,
                   const raft::device_matrix<T>& B,
                   raft::device_matrix<T>& C,
                   const raft::core::bitset& mask)
{
  cusparseSpMatDescr_t mask;

  mask.to_csr(mask, )
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatecsr(&matA,
                                                              nrows_,
                                                              ncols_,
                                                              nnz_,
                                                              const_cast<index_type*>(row_offsets_),
                                                              const_cast<index_type*>(col_indices_),
                                                              const_cast<value_type*>(values_)));
}

}  // namespace detail
}  // namespace linalg
}  // namespace raft