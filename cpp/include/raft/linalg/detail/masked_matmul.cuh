#pragma once

#include <raft/core/bitmap.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/handle.hpp>
#include <raft/linalg/matrix_mul.hpp>

namespace raft {
namespace linalg {
namespace detail {

template <typename T>
__global__ void masked_matmul_kernel(const T* A,
                                     size_t pitch_a,
                                     const T* B,
                                     size_t pitch_b,
                                     T* C,
                                     size_t pitch_c,
                                     const bool* mask,
                                     size_t pitch_mask,
                                     size_t m,
                                     size_t n,
                                     size_t k)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    T sum = 0;
    if (mask[row * pitch_mask + col]) {
      for (int e = 0; e < k; ++e) {
        sum += A[row * pitch_a + e] * B[e * pitch_b + col];
      }
    }
    C[row * pitch_c + col] = sum;
  }
}

template <typename T>
void masked_matmul(const raft::handle_t& handle,
                   const raft::device_matrix<T>& A,
                   const raft::device_matrix<T>& B,
                   raft::device_matrix<T>& C,
                   const raft::core::bitmap& mask)
{
  // Ensure A, B, and C are compatible for multiplication
  size_t m = A.rows();
  size_t k = A.cols();
  size_t n = B.cols();

  if (k != B.rows() || m != C.rows() || n != C.cols()) {
    // Handle error: incompatible dimensions
    return;
  }

  size_t pitch_a    = A.get_pitch();
  size_t pitch_b    = B.get_pitch();
  size_t pitch_c    = C.get_pitch();
  size_t pitch_mask = mask.get_pitch();  // Assuming mask has the same dimensions as C

  dim3 block(16, 16);  // block size (can be tuned for performance)
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

  masked_matmul_kernel<<<grid, block>>>(
    A.data(), pitch_a, B.data(), pitch_b, C.data(), pitch_c, mask.data(), pitch_mask, m, n, k);
  // Check for errors and synchronize
}

}  // namespace detail
}  // namespace linalg
}  // namespace raft