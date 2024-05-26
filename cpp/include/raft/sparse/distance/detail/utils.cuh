/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <raft/core/math.hpp>
#include <raft/distance/distance_types.hpp>

#include <cub/cub.cuh>

namespace raft {
namespace sparse {
namespace distance {
namespace detail {

/**
 * Computes the maximum number of columns that can be stored
 * in shared memory in dense form with the given block size
 * and precision.
 * @return the maximum number of columns that can be stored in smem
 */
template <typename value_idx, typename value_t, int tpb = 1024>
inline int max_cols_per_block()
{
  // max cols = (total smem available - cub reduction smem)
  return (raft::getSharedMemPerBlock() - ((tpb / raft::warp_size()) * sizeof(value_t))) /
         sizeof(value_t);
}

template <typename value_idx, typename value_t, typename expansion_f>
RAFT_KERNEL epilogue_on_csr_kernel(value_t* __restrict__ compressed_C,
                                   const value_idx* __restrict__ rows,
                                   const value_idx* __restrict__ cols,
                                   const value_t* __restrict__ Q_sq_norms,
                                   const value_t* __restrict__ R_sq_norms,
                                   value_idx nnz,
                                   expansion_f expansion_func)
{
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= nnz) return;
  const value_idx i = rows[tid];
  const value_idx j = cols[tid];

  compressed_C[tid] = expansion_func(compressed_C[tid], Q_sq_norms[i], R_sq_norms[j]);
}

template <typename value_idx, typename value_t, int tpb = 256>
void epilogue_on_csr(raft::resources const& handle,
                     value_t* compressed_C,
                     const value_idx nnz,
                     const value_idx* rows,
                     const value_idx* cols,
                     const value_t* Q_sq_norms,
                     const value_t* R_sq_norms,
                     raft::distance::DistanceType metric)
{
  auto stream = resource::get_cuda_stream(handle);

  int blocks = raft::ceildiv<size_t>((size_t)nnz, tpb);
  if (metric == raft::distance::DistanceType::L2Expanded) {
    epilogue_on_csr_kernel<<<blocks, tpb, 0, stream>>>(
      compressed_C,
      rows,
      cols,
      Q_sq_norms,
      R_sq_norms,
      nnz,
      [] __device__ __host__(value_t dot, value_t q_norm, value_t r_norm) -> value_t {
        return value_t(-2.0) * dot + q_norm + r_norm;
      });
  } else if (metric == raft::distance::DistanceType::L2SqrtExpanded) {
    epilogue_on_csr_kernel<<<blocks, tpb, 0, stream>>>(
      compressed_C,
      rows,
      cols,
      Q_sq_norms,
      R_sq_norms,
      nnz,
      [] __device__ __host__(value_t dot, value_t q_norm, value_t r_norm) -> value_t {
        return raft::sqrt(value_t(-2.0) * dot + q_norm + r_norm);
      });
  } else if (metric == raft::distance::DistanceType::CosineExpanded) {
    epilogue_on_csr_kernel<<<blocks, tpb, 0, stream>>>(
      compressed_C,
      rows,
      cols,
      Q_sq_norms,
      R_sq_norms,
      nnz,
      [] __device__ __host__(value_t dot, value_t q_norm, value_t r_norm) -> value_t {
        return value_t(1.0) - dot / (q_norm * r_norm);
      });
  }
}

template <typename value_idx, typename value_t>
__global__ void faster_dot_on_csr_kernel(value_t* __restrict__ dot,
                                         const value_idx* __restrict__ indptr,
                                         const value_idx* __restrict__ cols,
                                         const value_t* __restrict__ A,
                                         const value_t* __restrict__ B,
                                         const value_idx nnz,
                                         const value_idx n_rows,
                                         const value_idx dim)
{
    auto vec_id  = threadIdx.x;
    auto lane_id = threadIdx.x & 0x1f;

    extern __shared__ char smem[];
    value_t* s_A      = (value_t*)smem;
    value_idx cur_row = -1;

    for (int row = blockIdx.x; row < n_rows; row += gridDim.x) {
        for (int dot_id = blockIdx.y + indptr[row]; dot_id < indptr[row + 1]; dot_id += gridDim.y) {
            if (dot_id >= nnz) { return; }
            const value_idx col               = cols[dot_id] * dim;
            const value_t* __restrict__ B_col = B + col;

            if (threadIdx.x == 0) { dot[dot_id] = 0.0; }
            __syncthreads();

            if (cur_row != row) {
                for (value_idx k = vec_id; k < dim; k += blockDim.x) {
                    s_A[k] = A[row * dim + k];
                }
                cur_row = row;
            }

            value_t l_dot_ = 0.0;
            for (value_idx k = vec_id; k < dim; k += blockDim.x) {
                asm("prefetch.global.L2 [%0];" ::"l"(B_col + k + blockDim.x));
                l_dot_ += s_A[k] * __ldcg(B_col + k);
            }
            l_dot_ += __shfl_down_sync(0xffffffff, l_dot_, 16);
            l_dot_ += __shfl_down_sync(0xffff, l_dot_, 8);
            l_dot_ += __shfl_down_sync(0xff, l_dot_, 4);
            l_dot_ += __shfl_down_sync(0xf, l_dot_, 2);
            l_dot_ += __shfl_down_sync(0x3, l_dot_, 1);

            if (lane_id == 0) { atomicAdd_block(dot + dot_id, l_dot_); }
        }
    }
}

template <typename value_idx, typename value_t>
void faster_dot_on_csr(raft::resources const& handle,
                       value_t* dot,
                       const value_idx nnz,
                       const value_idx* indptr,
                       const value_idx* cols,
                       const value_t* A,
                       const value_t* B,
                       const value_idx n_rows,
                       const value_idx dim)
{
  if (nnz == 0 || n_rows == 0) return;

  auto stream = resource::get_cuda_stream(handle);

  constexpr value_idx MAX_ROW_PER_ITER = 500;
  int dev_id, sm_count, blocks_per_sm;

  const int smem_size = dim * sizeof(value_t);
  cudaGetDevice(&dev_id);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);

  if (dim < 128) {
    constexpr int tpb = 64;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, faster_dot_on_csr_kernel<value_idx, value_t>, tpb, smem_size);
    auto block_x = std::min(n_rows, MAX_ROW_PER_ITER);
    auto block_y =
      (std::min(value_idx(blocks_per_sm * sm_count * 16), nnz) + block_x - 1) / block_x;
    dim3 blocks(block_x, block_y, 1);

    faster_dot_on_csr_kernel<value_idx, value_t>
      <<<blocks, tpb, smem_size, stream>>>(dot, indptr, cols, A, B, nnz, n_rows, dim);

  } else if (dim < 256) {
    constexpr int tpb = 128;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, faster_dot_on_csr_kernel<value_idx, value_t>, tpb, smem_size);
    auto block_x = std::min(n_rows, MAX_ROW_PER_ITER);
    auto block_y =
      (std::min(value_idx(blocks_per_sm * sm_count * 16), nnz) + block_x - 1) / block_x;
    dim3 blocks(block_x, block_y, 1);

    faster_dot_on_csr_kernel<value_idx, value_t>
      <<<blocks, tpb, smem_size, stream>>>(dot, indptr, cols, A, B, nnz, n_rows, dim);
  } else if (dim < 512) {
    constexpr int tpb = 256;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, faster_dot_on_csr_kernel<value_idx, value_t>, tpb, smem_size);
    auto block_x = std::min(n_rows, MAX_ROW_PER_ITER);
    auto block_y =
      (std::min(value_idx(blocks_per_sm * sm_count * 16), nnz) + block_x - 1) / block_x;
    dim3 blocks(block_x, block_y, 1);

    faster_dot_on_csr_kernel<value_idx, value_t>
      <<<blocks, tpb, smem_size, stream>>>(dot, indptr, cols, A, B, nnz, n_rows, dim);
  } else {
    constexpr int tpb = 512;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, faster_dot_on_csr_kernel<value_idx, value_t>, tpb, smem_size);
    auto block_x = std::min(n_rows, MAX_ROW_PER_ITER);
    auto block_y =
      (std::min(value_idx(blocks_per_sm * sm_count * 16), nnz) + block_x - 1) / block_x;
    dim3 blocks(block_x, block_y, 1);

    faster_dot_on_csr_kernel<value_idx, value_t>
      <<<blocks, tpb, smem_size, stream>>>(dot, indptr, cols, A, B, nnz, n_rows, dim);
  }

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace detail
}  // namespace distance
}  // namespace sparse
}  // namespace raft
