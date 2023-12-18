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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cusparse_handle.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>

namespace raft {
namespace sparse {
namespace linalg {
namespace detail {

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
 * @tparam NZType Type of C
 *
 * @param[in] handle raft resource handle
 * @param[in] trans_a transpose operation for A
 * @param[in] trans_b transpose operation for B
 * @param[in] is_row_major data layout of A,B
 * @param[in] alpha scalar pointer
 * @param[in] descr_a input dense descriptor
 * @param[in] descr_b input dense descriptor
 * @param[in] beta scalar pointer
 * @param[out] descr_c output sparse descriptor
 */
template <typename ValueType>
void sddmm(raft::resources const& handle,
           const bool trans_a,
           const bool trans_b,
           const bool is_row_major,
           const ValueType* alpha,
           cusparseDnMatDescr_t& descr_a,
           cusparseDnMatDescr_t& descr_b,
           const ValueType* beta,
           cusparseSpMatDescr_t& descr_c)
{
  auto opA = trans_a ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto opB = trans_b ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto alg = CUSPARSE_SDDMM_ALG_DEFAULT;
  size_t bufferSize;
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsesddmm_bufferSize(resource::get_cusparse_handle(handle),
                                                   opA,
                                                   opB,
                                                   alpha,
                                                   descr_a,
                                                   descr_b,
                                                   beta,
                                                   descr_c,
                                                   alg,
                                                   &bufferSize,
                                                   resource::get_cuda_stream(handle)));

  raft::interruptible::synchronize(resource::get_cuda_stream(handle));

  rmm::device_uvector<ValueType> tmp(bufferSize, resource::get_cuda_stream(handle));

  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesddmm(resource::get_cusparse_handle(handle),
                                                        opA,
                                                        opB,
                                                        alpha,
                                                        descr_a,
                                                        descr_b,
                                                        beta,
                                                        descr_c,
                                                        alg,
                                                        tmp.data(),
                                                        resource::get_cuda_stream(handle)));
}

}  // end namespace detail
}  // end namespace linalg
}  // end namespace sparse
}  // end namespace raft
