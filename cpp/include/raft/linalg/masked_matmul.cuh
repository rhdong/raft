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

#include "detail/gemm.hpp"
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/input_validation.hpp>

namespace raft {
namespace linalg {

template <typename ValueType,
          typename IndexType,
          typename LayoutPolicyX,
          typename LayoutPolicyY,
          typename LayoutPolicyZ,
          typename ScalarIdxType  = std::uint32_t,
          typename ScalarViewType = raft::host_scalar_view<ValueType, ScalarIdxType>,
          typename                = std::enable_if_t<std::disjunction_v<
            std::is_same<ScalarViewType, raft::host_scalar_view<ValueType, ScalarIdxType>>,
            std::is_same<ScalarViewType, raft::device_scalar_view<ValueType, ScalarIdxType>>>>>
void masked_matmul(raft::resources const& handle,
                   raft::device_matrix_view<ValueType, IndexType, LayoutPolicyX> x,
                   raft::device_matrix_view<ValueType, IndexType, LayoutPolicyY> y,
                   raft::device_csr_matrix_view<ValueType, IndexType, LayoutPolicyZ> z,
                   std::optional<ScalarViewType> alpha = std::nullopt,
                   std::optional<ScalarViewType> beta  = std::nullopt)
{
  RAFT_EXPECTS(raft::is_row_or_column_major(x), "X is not contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(y), "Y is not contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(z), "Z is not contiguous");

  RAFT_EXPECTS(x.extent(0) == z.extent(0), "Number of rows of X and Z should be equal");
  RAFT_EXPECTS(y.extent(1) == z.extent(1), "Number of columns of Y and Z should be equal");
  RAFT_EXPECTS(x.extent(1) == y.extent(0), "Number of columns of X and rows of Y should be equal");

  constexpr auto is_x_col_major =
    std::is_same_v<typename decltype(x)::layout_type, raft::col_major>;
  constexpr auto is_y_col_major =
    std::is_same_v<typename decltype(y)::layout_type, raft::col_major>;
  constexpr auto is_z_col_major =
    std::is_same_v<typename decltype(z)::layout_type, raft::col_major>;

  constexpr auto device_mode =
    std::is_same_v<ScalarViewType, raft::device_scalar_view<ValueType, ScalarIdxType>>;

  ValueType alpha_value = 1;
  ValueType beta_value  = 0;

  auto alpha_device = raft::make_device_scalar(handle, alpha_value);
  auto beta_device  = raft::make_device_scalar(handle, beta_value);

  auto alpha_host = raft::make_host_scalar(alpha_value);
  auto beta_host  = raft::make_host_scalar(beta_value);

  if constexpr (device_mode) {
    if (!alpha) { alpha = alpha_device.view(); }
    if (!beta) { beta = beta_device.view(); }
  } else {
    if (!alpha) { alpha = alpha_host.view(); }
    if (!beta) { beta = beta_host.view(); }
  }

  detail::masked_matmul<ValueType, device_mode>(handle,
                                                z.data_handle(),
                                                x.data_handle(),
                                                y.data_handle(),
                                                x.extent(0),
                                                y.extent(1),
                                                x.extent(1),
                                                is_z_col_major,
                                                is_x_col_major,
                                                is_y_col_major,
                                                resource::get_cuda_stream(handle),
                                                alpha.value().data_handle(),
                                                beta.value().data_handle());
}

}  // end namespace linalg
}  // end namespace raft
