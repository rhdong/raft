/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cooperative_groups.h>

#include <raft/core/detail/mdspan_util.cuh>  // detail::popc
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/convert/detail/adj_to_csr.cuh>

#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

namespace raft {
namespace sparse {
namespace convert {
namespace detail {

// Threads per block in calc_nnz_by_rows_kernel.
static const constexpr int calc_nnz_by_rows_tpb = 256;

template <typename bitmap_t, typename index_t, typename nnz_t>
RAFT_KERNEL __launch_bounds__(calc_nnz_by_rows_tpb) calc_nnz_by_rows_kernel(const bitmap_t* bitmap,
                                                                            index_t num_rows,
                                                                            index_t num_cols,
                                                                            index_t bitmap_num,
                                                                            nnz_t* nnz_per_row)
{
  index_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
  for (index_t idx = thread_idx; idx < bitmap_num; idx += blockDim.x * gridDim.x) {
    index_t start  = idx * sizeof(bitmap_t) * 8;
    index_t offset = 0;
    //     printf("bitmap_num=%d\n", bitmap_num);
    while (offset < sizeof(bitmap_t) * 8) {
      bitmap_t mask = ~bitmap_t(0u);
      index_t row   = (start + offset) / num_cols;

      index_t delta = min(static_cast<index_t>(sizeof(bitmap_t) * 8) - offset, num_cols);

      mask >>= offset;
      mask <<= offset;
      index_t end_bit = num_cols * (row + 1);
      if (start + offset + delta >= end_bit) {
        mask <<= (sizeof(bitmap_t) * 8 - (end_bit - start));
        mask >>= (sizeof(bitmap_t) * 8 - (end_bit - start));
        delta = end_bit - offset - start;
      }
      atomicAdd(nnz_per_row + row, static_cast<nnz_t>(raft::detail::popc(bitmap[idx] & mask)));
      offset += delta;
    }
  }
}

template <typename bitmap_t, typename index_t, typename nnz_t>
void calc_nnz_by_rows(raft::resources const& handle,
                      const bitmap_t* bitmap,
                      index_t num_rows,
                      index_t num_cols,
                      nnz_t* nnz_per_row)
{
  auto stream              = resource::get_cuda_stream(handle);
  const index_t total      = num_rows * num_cols;
  const index_t bitmap_num = raft::ceildiv(total, index_t(sizeof(bitmap_t) * 8));

  int dev_id, sm_count, blocks_per_sm;

  cudaGetDevice(&dev_id);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_sm, calc_nnz_by_rows_kernel<bitmap_t, index_t, nnz_t>, calc_nnz_by_rows_tpb, 0);

  index_t max_active_blocks = sm_count * blocks_per_sm;
  auto grid = std::min(max_active_blocks, raft::ceildiv(bitmap_num, index_t(calc_nnz_by_rows_tpb)));
  auto block = calc_nnz_by_rows_tpb;

  calc_nnz_by_rows_kernel<bitmap_t, index_t, nnz_t>
    <<<grid, block, 0, stream>>>(bitmap, num_rows, num_cols, bitmap_num, nnz_per_row);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename value_t>
__device__ inline value_t warp_exclusive(value_t value)
{
  int lane_id           = threadIdx.x & 0x1f;
  value_t shifted_value = __shfl_up_sync(0xffffffff, value, 1, warpSize);
  if (lane_id == 0) shifted_value = 0;

  value_t sum = shifted_value;

  for (int i = 1; i < warpSize; i *= 2) {
    value_t n = __shfl_up_sync(0xffffffff, sum, i, warpSize);
    if (lane_id >= i) { sum += n; }
  }
  return sum;
}

// Threads per block in fill_indices_by_rows_kernel.
static const constexpr int fill_indices_by_rows_tpb = 32;

template <typename bitmap_t, typename index_t>
RAFT_KERNEL __launch_bounds__(fill_indices_by_rows_tpb)
  fill_indices_by_rows_kernel(const bitmap_t* bitmap,
                              const index_t* indptr,
                              index_t num_rows,
                              index_t num_cols,
                              index_t bitmap_num,
                              index_t* indices)
{
  constexpr bitmap_t FULL_MASK      = ~bitmap_t(0u);
  constexpr bitmap_t ONE            = bitmap_t(1u);
  constexpr index_t BITS_PER_BITMAP = sizeof(bitmap_t) * 8;

  int lane_id = threadIdx.x & 0x1f;

  for (index_t row = blockIdx.x; row < num_rows; row += gridDim.x) {
    index_t offset = 0;
    index_t g_sum  = 0;
    index_t s_bit  = row * num_cols;
    index_t e_bit  = s_bit + num_cols;


    while (offset < num_cols) {
      index_t bitmap_idx = lane_id + (s_bit + offset) / BITS_PER_BITMAP;
      bitmap_t l_bitmap = bitmap_t(0);

      if(bitmap_idx * BITS_PER_BITMAP < e_bit) {
        l_bitmap  = bitmap[bitmap_idx];
      }

      if (s_bit > bitmap_idx * BITS_PER_BITMAP) {
        l_bitmap >>= (s_bit - bitmap_idx * BITS_PER_BITMAP);
        l_bitmap <<= (s_bit - bitmap_idx * BITS_PER_BITMAP);
      }

      if ((bitmap_idx + 1) * BITS_PER_BITMAP > e_bit) {
        l_bitmap <<= ((bitmap_idx + 1) * BITS_PER_BITMAP - e_bit);
        l_bitmap >>= ((bitmap_idx + 1) * BITS_PER_BITMAP - e_bit);
      }

      index_t l_sum = warp_exclusive(static_cast<index_t>(raft::detail::popc(l_bitmap)));

      for (int i = 0; i < BITS_PER_BITMAP; i++) {
        if(l_bitmap & (ONE << i)) {
          indices[indptr[row] + g_sum + l_sum] = offset - (s_bit % BITS_PER_BITMAP) - lane_id * BITS_PER_BITMAP + i;
          l_sum++;
          printf("row=%d, lane_id=%d, indptr[row]=%d, g_sum=%d, l_sum=%d, offset=%d, i=%d, l_bitmap=%d, s_bit=%d, r=%d\n",
                 row, lane_id, indptr[row], g_sum, l_sum, offset, i, l_bitmap, s_bit,
                 offset - (s_bit % BITS_PER_BITMAP) - lane_id * BITS_PER_BITMAP + i);
        }
      }
      offset += BITS_PER_BITMAP * warpSize;
      g_sum = __shfl_sync(0xffffffff, g_sum + l_sum, warpSize - 1);
    }
  }
}

template <typename bitmap_t, typename index_t>
void fill_indices_by_rows(raft::resources const& handle,
                          const bitmap_t* bitmap,
                          const index_t* indptr,
                          index_t num_rows,
                          index_t num_cols,
                          index_t* indices)
{
  auto stream              = resource::get_cuda_stream(handle);
  const index_t total      = num_rows * num_cols;
  const index_t bitmap_num = raft::ceildiv(total, index_t(sizeof(bitmap_t) * 8));

  int dev_id, sm_count, blocks_per_sm;

  cudaGetDevice(&dev_id);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_sm, fill_indices_by_rows_kernel<bitmap_t, index_t>, fill_indices_by_rows_tpb, 0);

  index_t max_active_blocks = sm_count * blocks_per_sm;
  auto grid                 = std::min(max_active_blocks, num_rows);
  auto block                = fill_indices_by_rows_tpb;

  fill_indices_by_rows_kernel<bitmap_t, index_t>
    <<<grid, block, 0, stream>>>(bitmap, indptr, num_rows, num_cols, bitmap_num, indices);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

// // Threads per block in fill_indices_by_rows_kernel.
// static const constexpr int fill_indices_by_rows_tpb = 32;
//
// template <typename bitmap_t, typename index_t>
// RAFT_KERNEL __launch_bounds__(fill_indices_by_rows_tpb)
//   fill_indices_by_rows_kernel(const bitmap_t* bitmap,
//                               const index_t* indptr,
//                               index_t num_rows,
//                               index_t num_cols,
//                               index_t bitmap_num,
//                               index_t* indices)
// {
//   constexpr bitmap_t FULL_MASK = ~bitmap_t(0u);
//   constexpr bitmap_t ONE       = bitmap_t(1u);
//
//   for (index_t row = blockIdx.x; row < num_rows; row += gridDim.x) {
//     index_t offset = 0;
//     index_t accum  = 0;
//     while (offset < num_cols) {
//       index_t global_bits = row * num_cols + offset + threadIdx.x;
//       index_t bitmap_idx  = global_bits / (sizeof(bitmap_t) * 8);
//       index_t bitmap_bit  = global_bits - bitmap_idx * (sizeof(bitmap_t) * 8);
//
//       bitmap_t set_one =
//         (global_bits >= (row + 1) * num_cols) ? 0 : (bitmap[bitmap_idx] & (ONE << bitmap_bit));
//       bitmap_t set_bits = __ballot_sync(FULL_MASK, set_one);
//
//       if (set_one) {
//         indices[indptr[row] + accum +
//                 static_cast<index_t>(raft::detail::popc(
//                   set_bits & (FULL_MASK >> (sizeof(bitmap_t) * 8 - threadIdx.x))))] =
//           offset + threadIdx.x;
//       }
//
//       if constexpr (sizeof(bitmap_t) == 8) {
//         set_one = (global_bits >= (row + 1) * num_cols)
//                     ? 0
//                     : (bitmap[bitmap_idx] & (ONE << (bitmap_bit + warpSize)));
//         set_bits |= (static_cast<bitmap_t>(__ballot_sync(FULL_MASK, set_one)) << warpSize);
//         if (set_one) {
//           indices[indptr[row] + accum +
//                   static_cast<index_t>(raft::detail::popc(
//                     set_bits & (FULL_MASK >> (sizeof(bitmap_t) * 8 - warpSize - threadIdx.x))))]
//                     =
//             offset + threadIdx.x + warpSize;
//         }
//       }
//
//       offset += sizeof(bitmap_t) * 8;
//       accum += raft::detail::popc(set_bits);
//     }
//   }
// }
//
// template <typename bitmap_t, typename index_t>
// void fill_indices_by_rows(raft::resources const& handle,
//                           const bitmap_t* bitmap,
//                           const index_t* indptr,
//                           index_t num_rows,
//                           index_t num_cols,
//                           index_t* indices)
// {
//   auto stream              = resource::get_cuda_stream(handle);
//   const index_t total      = num_rows * num_cols;
//   const index_t bitmap_num = raft::ceildiv(total, index_t(sizeof(bitmap_t) * 8));
//
//   int dev_id, sm_count, blocks_per_sm;
//
//   cudaGetDevice(&dev_id);
//   cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
//   cudaOccupancyMaxActiveBlocksPerMultiprocessor(
//     &blocks_per_sm, fill_indices_by_rows_kernel<bitmap_t, index_t>, fill_indices_by_rows_tpb, 0);
//
//   index_t max_active_blocks = sm_count * blocks_per_sm;
//   auto grid                 = std::min(max_active_blocks, num_rows);
//   auto block                = fill_indices_by_rows_tpb;
//
//   fill_indices_by_rows_kernel<bitmap_t, index_t>
//     <<<grid, block, 0, stream>>>(bitmap, indptr, num_rows, num_cols, bitmap_num, indices);
//   RAFT_CUDA_TRY(cudaPeekAtLastError());
// }

template <typename bitmap_t, typename index_t, typename nnz_t>
void bitmap_to_csr(raft::resources const& handle,
                   const bitmap_t* bitmap,
                   index_t num_rows,
                   index_t num_cols,
                   nnz_t nnz,
                   index_t* indptr,
                   index_t* indices)
{
  const index_t total = num_rows * num_cols;
  if (total == 0) { return; }

  auto thrust_policy = resource::get_thrust_policy(handle);
  auto stream        = resource::get_cuda_stream(handle);

  RAFT_CUDA_TRY(cudaMemsetAsync(indptr, 0, (num_rows + 1) * sizeof(index_t), stream));

  calc_nnz_by_rows(handle, bitmap, num_rows, num_cols, indptr);
  thrust::exclusive_scan(thrust_policy, indptr, indptr + num_rows + 1, indptr);
  fill_indices_by_rows(handle, bitmap, indptr, num_rows, num_cols, indices);
}

};  // end NAMESPACE detail
};  // end NAMESPACE convert
};  // end NAMESPACE sparse
};  // end NAMESPACE raft
