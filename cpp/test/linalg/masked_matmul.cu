/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

#include "../test_utils.cuh"
#include "masked_matmul.cuh"
#include <gtest/gtest.h>
#include <raft/core/bitmap.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>

namespace raft {
namespace linalg {

template <typename T>
bool compareDeviceMatrixWithHost(const raft::device_matrix<T>& d_matrix,
                                 const std::vector<T>& h_expected,
                                 size_t m,
                                 size_t n)
{
  std::vector<T> h_result(m * n);
  cudaMemcpy(h_result.data(), d_matrix.data(), m * n * sizeof(T), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < m * n; ++i) {
    if (h_result[i] != h_expected[i]) { return false; }
  }
  return true;
}

class MaskedMatmulTest : public ::testing::Test {
 protected:
  raft::handle_t handle;
};

TEST_F(MaskedMatmulTest, BasicMultiplication)
{
  // Define matrices A, B, and C
  const size_t m = 2, n = 2, k = 2;
  std::vector<float> h_A          = {1, 2, 3, 4};      // Host-side matrix A
  std::vector<float> h_B          = {5, 6, 7, 8};      // Host-side matrix B
  std::vector<float> h_C_expected = {19, 22, 43, 50};  // Expected result of A*B

  // Create device matrices
  raft::device_matrix<float> d_A(m, k, handle.get_stream());Í
  raft::device_matrix<float> d_B(k, n, handle.get_stream());
  raft::device_matrix<float> d_C(m, n, handle.get_stream());

  // Copy data to device
  cudaMemcpy(d_A.data(), h_A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B.data(), h_B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice);

  // Create a mask (all true for basic multiplication)
  raft::core::bitmap mask(m, n, true, handle.get_stream());

  // Call the masked matrix multiplication
  raft::linalg::masked_matmul(handle, d_A, d_B, d_C, mask);

  // Check if the result is as expected
  ASSERT_TRUE(compareDeviceMatrixWithHost(d_C, h_C_expected, m, n));
}

}  // end namespace linalg
}  // end namespace raft
