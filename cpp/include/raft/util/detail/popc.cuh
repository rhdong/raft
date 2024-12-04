/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/core/detail/mdspan_util.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>

#include <thrust/reduce.h>

namespace raft::detail {

// Threads per block in popc_tpb.
static const constexpr int popc_tpb = 256;

template <typename bitset_t, typename index_t>
RAFT_KERNEL __launch_bounds__(popc_tpb)
  segment_popc_kernel(typename std::remove_const_t<bitset_t>* bitset,
                      index_t total,
                      index_t* seg_nnz,
                      index_t bits_per_seg)
{
  using mutable_bitset_t = typename std::remove_const_t<bitset_t>;
  using popc_t =
    std::conditional_t<sizeof(mutable_bitset_t) <= sizeof(uint32_t), uint32_t, uint64_t>;

  using BlockReduce = cub::BlockReduce<index_t, popc_tpb>;

  __shared__ typename BlockReduce::TempStorage reduce_storage;

  constexpr index_t BITS_PER_BITMAP = sizeof(bitset_t) * 8;

  const auto tid = threadIdx.x;
  const auto seg = blockIdx.x;

  size_t s_bit = seg * bits_per_seg;
  size_t e_bit = min(s_bit + bits_per_seg, size_t(total));

  index_t l_sum = 0;
  index_t g_sum = 0;

  size_t bitset_idx = s_bit / BITS_PER_BITMAP;

  for (size_t bit_idx = s_bit; bit_idx < e_bit; bit_idx += BITS_PER_BITMAP * blockDim.x) {
    mutable_bitset_t l_bitset = 0;
    bitset_idx                = bit_idx / BITS_PER_BITMAP + tid;

    index_t remaining_bits = min(BITS_PER_BITMAP, index_t(e_bit - bitset_idx * BITS_PER_BITMAP));

    if (bitset_idx * BITS_PER_BITMAP < e_bit) { l_bitset = bitset[bitset_idx]; }

    if (remaining_bits < BITS_PER_BITMAP) {
      l_bitset &= ((mutable_bitset_t(1) << remaining_bits) - 1);
    }
    l_sum += raft::detail::popc(static_cast<popc_t>(l_bitset));
  }
  g_sum = BlockReduce(reduce_storage).Reduce(l_sum, cub::Sum());

  if (tid == 0) { seg_nnz[seg] = g_sum; }
}

template <typename bitset_t, typename index_t>
void segment_popc(raft::resources const& handle,
                  bitset_t* bitset,
                  index_t total,
                  index_t* seg_nnz,
                  size_t& sub_nnz_size,
                  index_t& bits_per_seg)
{
  if (sub_nnz_size == 0) {
    bits_per_seg = popc_tpb * sizeof(index_t) * 8 * 8;
    sub_nnz_size = (total + bits_per_seg - 1) / bits_per_seg;
    return;
  }

  auto stream = resource::get_cuda_stream(handle);
  auto grid   = sub_nnz_size;
  auto block  = popc_tpb;

  segment_popc_kernel<typename std::remove_const_t<bitset_t>, index_t>
    <<<grid, block, 0, stream>>>(bitset, total, seg_nnz, bits_per_seg);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * @brief Count the number of bits that are set to 1 in a vector.
 *
 * @tparam value_t the value type of the vector.
 * @tparam index_t the index type of vector and scalar.
 *
 * @param[in] res RAFT handle for managing expensive resources
 * @param[in] values Device vector view containing the values to be processed.
 * @param[in] max_len Maximum number of bits to count.
 * @param[out] counter Device scalar view to store the number of bits that are set to 1.
 */
template <typename value_t, typename index_t>
void popc(const raft::resources& res,
          device_vector_view<const value_t, index_t> values,
          raft::host_scalar_view<const index_t, index_t> max_len,
          raft::device_scalar_view<index_t> counter)
{
  auto stream        = resource::get_cuda_stream(res);
  auto thrust_policy = resource::get_thrust_policy(res);

  size_t sub_nnz_size  = 0;
  index_t bits_per_seg = 0;

  // Get buffer size and number of bits per each segment
  segment_popc(res,
               const_cast<typename std::remove_const_t<value_t>*>(values.data_handle()),
               max_len[0],
               static_cast<index_t*>(nullptr),
               sub_nnz_size,
               bits_per_seg);

  rmm::device_async_resource_ref device_memory = resource::get_workspace_resource(res);
  rmm::device_uvector<index_t> sub_nnz(sub_nnz_size, stream, device_memory);

  segment_popc(res,
               const_cast<typename std::remove_const_t<value_t>*>(values.data_handle()),
               max_len[0],
               sub_nnz.data(),
               sub_nnz_size,
               bits_per_seg);

  index_t sum = thrust::reduce(thrust_policy,
                               sub_nnz.data(),
                               sub_nnz.data() + sub_nnz_size,
                               index_t{0},
                               thrust::plus<index_t>());

  cudaMemcpyAsync(counter.data_handle(), &sum, sizeof(index_t), cudaMemcpyHostToDevice, stream);
}

}  // end namespace raft::detail
