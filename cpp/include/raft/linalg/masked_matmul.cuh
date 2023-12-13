#pragma once

#include <raft/core/bitmap.hpp>
#include <raft/core/device_matrix.hpp>
#include <raft/handle.hpp>
#include <raft/linalg/matrix_mul.hpp>

namespace raft {
namespace linalg {

template <typename T>
void masked_matmul(const raft::handle_t& handle,
                   const raft::device_matrix<T>& A,
                   const raft::device_matrix<T>& B,
                   raft::device_matrix<T>& C,
                   const raft::core::bitmap& mask)
{
  detail::masked_matmul(handle, A, B, C, mask);
}

}  // namespace linalg
}  // namespace raft