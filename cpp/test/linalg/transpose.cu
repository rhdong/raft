/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_fp16.h>

#include <gtest/gtest.h>

namespace raft {
namespace linalg {

template <typename T, typename IdxT>
__global__ void dump_array_kernel(T* array, IdxT size, int id)
{
  printf("device: %d\n", id);
  for (IdxT i = 0; i < size; i++) {
    printf("%d\t%f\n", int(i), float(array[i]));
  }
  printf("\n");
}

template <typename T>
void dump_vector(const T* vec, size_t size, const std::string& name)
{
  std::cout << "Dumping vector " << name << " (" << size << " elements):" << std::endl;
  for (size_t i = 0; i < size; ++i) {
    std::cout << name << "[" << i << "] = " << float(vec[i]) << std::endl;
  }
}

template <typename T>
void initialize_array(T* data_h, size_t size)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  for (size_t i = 0; i < size; ++i) {
    if constexpr (std::is_same_v<T, half>) {
      data_h[i] = __float2half(static_cast<float>(dis(gen)));
    } else {
      data_h[i] = static_cast<T>(dis(gen));
    }
  }
}

template <typename T>
void cpu_transpose_col_major(const T* input, T* output, int rows, int cols)
{
  for (int i = 0; i < cols; ++i) {
    for (int j = 0; j < rows; ++j) {
      output[j * cols + i] = input[i * rows + j];
    }
  }
}

template <typename T>
void cpu_transpose_row_major(const T* input, T* output, int rows, int cols)
{
  cpu_transpose_col_major(input, output, cols, rows);
}

bool validate_half(const half* h_ref, const half* h_result, half tolerance, int len)
{
  bool success = true;
  for (int i = 0; i < len; ++i) {
    if (raft::abs(__half2float(h_result[i]) - __half2float(h_ref[i])) >= __half2float(tolerance)) {
      success = false;
      break;
    }
    if (!success) break;
  }
  return success;
}

template <typename T>
struct TransposeInputs {
  T tolerance;
  int n_row;
  int n_col;
  unsigned long long int seed;
};

namespace transpose_regular_test {

template <typename T>
class TransposeTest : public ::testing::TestWithParam<TransposeInputs<T>> {
 public:
  TransposeTest()
    : params(::testing::TestWithParam<TransposeInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      data(params.n_row * params.n_col, stream),
      data_trans_ref(params.n_row * params.n_col, stream),
      data_trans(params.n_row * params.n_col, stream)
  {
  }

 protected:
  void SetUp() override
  {
    int len = params.n_row * params.n_col;
    std::unique_ptr<T[]> data_h(new T[len]);
    std::unique_ptr<T[]> data_ref_h(new T[len]);

    initialize_array(data_h.get(), len);

    cpu_transpose_col_major(data_h.get(), data_ref_h.get(), params.n_row, params.n_col);

    raft::update_device(data.data(), data_h.get(), len, stream);
    raft::update_device(data_trans_ref.data(), data_ref_h.get(), len, stream);

    transpose(handle, data.data(), data_trans.data(), params.n_row, params.n_col, stream);
    if (params.n_row == params.n_col) { transpose(data.data(), params.n_col, stream); }
    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  TransposeInputs<T> params;
  rmm::device_uvector<T> data, data_trans, data_trans_ref;
};

const std::vector<TransposeInputs<float>> inputsf2 = {{0.1f, 3, 3, 1234ULL},
                                                      {0.1f, 3, 4, 1234ULL},
                                                      {0.1f, 300, 300, 1234ULL},
                                                      {0.1f, 300, 4100, 1234ULL},
                                                      {0.1f, 1, 13000, 1234ULL},
                                                      {0.1f, 3, 1300001, 1234ULL}};

const std::vector<TransposeInputs<double>> inputsd2 = {{0.1f, 3, 3, 1234ULL},
                                                       {0.1f, 3, 4, 1234ULL},
                                                       {0.1f, 300, 300, 1234ULL},
                                                       {0.1f, 300, 4100, 1234ULL},
                                                       {0.1f, 1, 13000, 1234ULL},
                                                       {0.1f, 3, 1300001, 1234ULL}};

const std::vector<TransposeInputs<half>> inputsh2 = {{0.1f, 3, 3, 1234ULL},
                                                     {0.1f, 3, 4, 1234ULL},
                                                     {0.1f, 300, 300, 1234ULL},
                                                     {0.1f, 300, 4100, 1234ULL},
                                                     {0.1f, 1, 13000, 1234ULL},
                                                     {0.1f, 3, 1300001, 1234ULL}};

typedef TransposeTest<float> TransposeTestValF;
TEST_P(TransposeTestValF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(data_trans_ref.data(),
                                data_trans.data(),
                                params.n_row * params.n_col,
                                raft::CompareApproxAbs<float>(params.tolerance)));

  if (params.n_row == params.n_col) {
    ASSERT_TRUE(raft::devArrMatch(data_trans_ref.data(),
                                  data.data(),
                                  params.n_row * params.n_col,
                                  raft::CompareApproxAbs<float>(params.tolerance)));
  }
}

typedef TransposeTest<double> TransposeTestValD;
TEST_P(TransposeTestValD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(data_trans_ref.data(),
                                data_trans.data(),
                                params.n_row * params.n_col,
                                raft::CompareApproxAbs<double>(params.tolerance)));
  if (params.n_row == params.n_col) {
    ASSERT_TRUE(raft::devArrMatch(data_trans_ref.data(),
                                  data.data(),
                                  params.n_row * params.n_col,
                                  raft::CompareApproxAbs<double>(params.tolerance)));
  }
}

typedef TransposeTest<half> TransposeTestValH;
TEST_P(TransposeTestValH, Result)
{
  std::vector<half> data_trans_ref_h(params.n_row * params.n_col);
  std::vector<half> data_trans_h(params.n_row * params.n_col);
  std::vector<half> data_h(params.n_row * params.n_col);

  RAFT_CUDA_TRY(cudaMemcpyAsync(data_trans_ref_h.data(),
                                data_trans_ref.data(),
                                params.n_row * params.n_col * sizeof(half),
                                cudaMemcpyDeviceToHost,
                                stream));

  RAFT_CUDA_TRY(cudaMemcpyAsync(data_trans_h.data(),
                                data_trans.data(),
                                params.n_row * params.n_col * sizeof(half),
                                cudaMemcpyDeviceToHost,
                                stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(data_h.data(),
                                data.data(),
                                params.n_row * params.n_col * sizeof(half),
                                cudaMemcpyDeviceToHost,
                                stream));

  resource::sync_stream(handle, stream);

  ASSERT_TRUE(validate_half(
    data_trans_ref_h.data(), data_trans_h.data(), params.tolerance, params.n_row * params.n_col));

  if (params.n_row == params.n_col) {
    ASSERT_TRUE(validate_half(
      data_trans_ref_h.data(), data_h.data(), params.tolerance, params.n_row * params.n_col));
  }
}

INSTANTIATE_TEST_SUITE_P(TransposeTests, TransposeTestValF, ::testing::ValuesIn(inputsf2));
INSTANTIATE_TEST_SUITE_P(TransposeTests, TransposeTestValD, ::testing::ValuesIn(inputsd2));
INSTANTIATE_TEST_SUITE_P(TransposeTests, TransposeTestValH, ::testing::ValuesIn(inputsh2));
}  // namespace transpose_regular_test

namespace transpose_mdspan_test {
/**
 * We hide these functions in tests for now until we have a heterogeneous mdarray
 * implementation.
 */

/**
 * @brief Transpose a matrix. The output has same layout policy as the input.
 *
 * @tparam T Data type of input matrix elements.
 * @tparam LayoutPolicy Layout type of the input matrix. When layout is strided, it can
 *                      be a submatrix of a larger matrix. Arbitrary stride is not supported.
 *
 * @param[in] handle raft handle for managing expensive cuda resources.
 * @param[in] in     Input matrix.
 *
 * @return The transposed matrix.
 */
template <typename T, typename IndexType, typename LayoutPolicy>
[[nodiscard]] auto transpose(raft::resources const& handle,
                             device_matrix_view<T, IndexType, LayoutPolicy> in)
  -> std::enable_if_t<std::is_floating_point_v<T> &&
                        (std::is_same_v<LayoutPolicy, layout_c_contiguous> ||
                         std::is_same_v<LayoutPolicy, layout_f_contiguous>),
                      device_matrix<T, IndexType, LayoutPolicy>>
{
  auto out = make_device_matrix<T, IndexType, LayoutPolicy>(handle, in.extent(1), in.extent(0));
  ::raft::linalg::transpose(handle, in, out.view());
  return out;
}

/**
 * @brief Transpose a matrix. The output has same layout policy as the input.
 *
 * @tparam T Data type of input matrix elements.
 * @tparam LayoutPolicy Layout type of the input matrix. When layout is strided, it can
 *                      be a submatrix of a larger matrix. Arbitrary stride is not supported.
 *
 * @param[in] handle raft handle for managing expensive cuda resources.
 * @param[in] in     Input matrix.
 *
 * @return The transposed matrix.
 */
template <typename T, typename IndexType>
[[nodiscard]] auto transpose(raft::resources const& handle,
                             device_matrix_view<T, IndexType, layout_stride> in)
  -> std::enable_if_t<std::is_floating_point_v<T>, device_matrix<T, IndexType, layout_stride>>
{
  matrix_extent<size_t> exts{in.extent(1), in.extent(0)};
  using policy_type =
    typename raft::device_matrix<T, IndexType, layout_stride>::container_policy_type;
  policy_type policy{};

  RAFT_EXPECTS(in.stride(0) == 1 || in.stride(1) == 1, "Unsupported matrix layout.");
  if (in.stride(1) == 1) {
    // row-major submatrix
    std::array<size_t, 2> strides{in.extent(0), 1};
    auto layout = layout_stride::mapping<matrix_extent<size_t>>{exts, strides};
    raft::device_matrix<T, IndexType, layout_stride> out{handle, layout, policy};
    ::raft::linalg::transpose(handle, in, out.view());
    return out;
  } else {
    // col-major submatrix
    std::array<size_t, 2> strides{1, in.extent(1)};
    auto layout = layout_stride::mapping<matrix_extent<size_t>>{exts, strides};
    raft::device_matrix<T, IndexType, layout_stride> out{handle, layout, policy};
    ::raft::linalg::transpose(handle, in, out.view());
    return out;
  }
}
template <typename T, typename LayoutPolicy>
void test_transpose_with_mdspan(const TransposeInputs<T>& param)
{
  raft::resources handle;
  auto v = make_device_matrix<T, size_t, LayoutPolicy>(handle, param.n_row, param.n_col);
  T k{0};
  for (size_t i = 0; i < v.extent(0); ++i) {
    for (size_t j = 0; j < v.extent(1); ++j) {
      v(i, j) = k++;
    }
  }
  auto out = transpose(handle, v.view());
  static_assert(std::is_same_v<LayoutPolicy, typename decltype(out)::layout_type>);
  ASSERT_EQ(out.extent(0), v.extent(1));
  ASSERT_EQ(out.extent(1), v.extent(0));

  k = 0;
  for (size_t i = 0; i < out.extent(1); ++i) {
    for (size_t j = 0; j < out.extent(0); ++j) {
      ASSERT_EQ(out(j, i), k++);
    }
  }
}

const std::vector<TransposeInputs<float>> inputs_mdspan_f = {{0.1f, 3, 3, 1234ULL},
                                                             {0.1f, 3, 4, 1234ULL}};

TEST(TransposeTest, MDSpanFloat)
{
  for (const auto& p : inputs_mdspan_f) {
    test_transpose_with_mdspan<float, layout_c_contiguous>(p);
    test_transpose_with_mdspan<float, layout_f_contiguous>(p);
  }
}

template <typename T, typename LayoutPolicy>
void test_transpose_submatrix()
{
  raft::resources handle;
  auto v = make_device_matrix<T, size_t, LayoutPolicy>(handle, 32, 33);
  T k{0};
  size_t row_beg{3}, row_end{13}, col_beg{2}, col_end{11};
  for (size_t i = row_beg; i < row_end; ++i) {
    for (size_t j = col_beg; j < col_end; ++j) {
      v(i, j) = k++;
    }
  }

  auto vv     = v.view();
  auto submat = std::experimental::submdspan(
    vv, std::make_tuple(row_beg, row_end), std::make_tuple(col_beg, col_end));
  static_assert(std::is_same_v<typename decltype(submat)::layout_type, layout_stride>);

  auto out = transpose(handle, submat);
  ASSERT_EQ(out.extent(0), submat.extent(1));
  ASSERT_EQ(out.extent(1), submat.extent(0));

  k = 0;
  for (size_t i = 0; i < out.extent(1); ++i) {
    for (size_t j = 0; j < out.extent(0); ++j) {
      ASSERT_EQ(out(j, i), k++);
    }
  }
}

TEST(TransposeTest, SubMatrix)
{
  test_transpose_submatrix<float, layout_c_contiguous>();
  test_transpose_submatrix<double, layout_c_contiguous>();

  test_transpose_submatrix<float, layout_f_contiguous>();
  test_transpose_submatrix<double, layout_f_contiguous>();
}

}  // namespace transpose_mdspan_test
}  // end namespace linalg
}  // end namespace raft
