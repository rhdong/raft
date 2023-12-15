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

#include <raft/core/bitset.cuh>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/matrix_mul.cuh>

namespace raft {
namespace linalg {

template <typename T>
void masked_matmul(const raft::handle_t& handle,
                   const raft::device_matrix<T>& A,
                   const raft::device_matrix<T>& B,
                   raft::device_matrix<T>& C,
                   const raft::core::bitset& bitmap)
{
  detail::masked_matmul(handle, A, B, C, mask);
}

}  // namespace linalg
}  // namespace raft