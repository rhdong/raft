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

#include "../test_utils.cuh"

#include <raft/core/bitmap.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/linalg/init.cuh>
#include <raft/random/rng.cuh>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>

namespace raft::core {

struct test_spec_bitmap {
  uint64_t bitmap_len;

  uint64_t n_rows;
  uint64_t n_cols;
  uint64_t mask_len;
  uint64_t query_len;
};

auto operator<<(std::ostream& os, const test_spec_bitmap& ss) -> std::ostream&
{
  os << "bitmap{n_rows: " << ss.n_rows << ", n_cols: " << ss.n_cols << ", mask_len: " << ss.mask_len
     << ", query_len: " << ss.query_len << "}";
  return os;
}

template <typename bitmap_t, typename index_t>
void add_cpu_bitmap(std::vector<bitmap_t>& bitmap, const std::vector<index_t>& mask_idx)
{
  constexpr size_t bitmap_element_size = sizeof(bitmap_t) * 8;
  for (size_t i = 0; i < mask_idx.size(); i++) {
    auto idx = mask_idx[i];
    bitmap[idx / bitmap_element_size] &= ~(bitmap_t{1} << (idx % bitmap_element_size));
  }
}

template <typename bitmap_t, typename index_t>
void create_cpu_bitmap(std::vector<bitmap_t>& bitmap, const std::vector<index_t>& mask_idx)
{
  for (size_t i = 0; i < bitmap.size(); i++) {
    bitmap[i] = ~bitmap_t(0x00);
  }
  add_cpu_bitmap(bitmap, mask_idx);
}

template <typename bitmap_t, typename index_t>
void test_cpu_bitmap(const std::vector<bitmap_t>& bitmap,
                     const std::vector<index_t>& queries,
                     std::vector<uint8_t>& result)
{
  constexpr size_t bitmap_element_size = sizeof(bitmap_t) * 8;
  for (size_t i = 0; i < queries.size(); i++) {
    result[i] = uint8_t((bitmap[queries[i] / bitmap_element_size] &
                         (bitmap_t{1} << (queries[i] % bitmap_element_size))) != 0);
  }
}

template <typename bitmap_t>
void flip_cpu_bitmap(std::vector<bitmap_t>& bitmap)
{
  for (size_t i = 0; i < bitmap.size(); i++) {
    bitmap[i] = ~bitmap[i];
  }
}

template <typename bitmap_t, typename index_t>
class BitmapTest : public testing::TestWithParam<test_spec_bitmap> {
 protected:
  index_t static constexpr const bitmap_element_size = sizeof(bitmap_t) * 8;
  const test_spec_bitmap spec;
  std::vector<bitmap_t> bitmap_result;
  std::vector<bitmap_t> bitmap_ref;
  raft::resources res;

 public:
  explicit BitmapTest()
    : spec(testing::TestWithParam<test_spec_bitmap>::GetParam()),
      bitmap_result(raft::ceildiv(spec.n_rows * spec.n_cols, uint64_t(bitmap_element_size))),
      bitmap_ref(raft::ceildiv(spec.n_rows * spec.n_cols, uint64_t(bitmap_element_size)))
  {
  }

  void run()
  {
    auto stream = resource::get_cuda_stream(res);

    // generate input and mask
    raft::random::RngState rng(42);
    auto mask_device = raft::make_device_vector<index_t, index_t>(res, spec.mask_len);
    std::vector<index_t> mask_cpu(spec.mask_len);
    raft::random::uniformInt(
      res, rng, mask_device.view(), index_t(0), index_t(spec.n_rows * spec.n_cols));
    update_host(mask_cpu.data(), mask_device.data_handle(), mask_device.extent(0), stream);
    resource::sync_stream(res, stream);

    // calculate the results
    auto my_bitset = raft::core::bitset<bitmap_t, index_t>(
      res, raft::make_const_mdspan(mask_device.view()), index_t(spec.bitset_len));
    auto my_bitmap = raft::core::bitmap<bitmap_t, index_t>(
      my_bitset.view(), index_t(spec.n_rows), index_t(spec.n_cols));
    update_host(bitmap_result.data(), my_bitset.data(), my_bitset.size(), stream);

    // calculate the reference
    create_cpu_bitmap(bitmap_ref, mask_cpu);
    resource::sync_stream(res, stream);
    ASSERT_TRUE(hostVecMatch(bitmap_ref, bitmap_result, raft::Compare<bitmap_t>()));

    auto query_device  = raft::make_device_vector<index_t, index_t>(res, spec.query_len);
    auto result_device = raft::make_device_vector<uint8_t, index_t>(res, spec.query_len);
    auto query_cpu     = std::vector<index_t>(spec.query_len);
    auto result_cpu    = std::vector<uint8_t>(spec.query_len);
    auto result_ref    = std::vector<uint8_t>(spec.query_len);

    // Create queries and verify the test results
    raft::random::uniformInt(
      res, rng, query_device.view(), index_t(0), index_t(spec.n_rows * spec.n_cols));
    update_host(query_cpu.data(), query_device.data_handle(), query_device.extent(0), stream);
    my_bitmap.test(res, raft::make_const_mdspan(query_device.view()), result_device.view());
    update_host(result_cpu.data(), result_device.data_handle(), result_device.extent(0), stream);
    test_cpu_bitmap(bitmap_ref, query_cpu, result_ref);
    resource::sync_stream(res, stream);
    ASSERT_TRUE(hostVecMatch(result_cpu, result_ref, Compare<uint8_t>()));

    // Add more sample to the bitmap and re-test
    raft::random::uniformInt(
      res, rng, mask_device.view(), index_t(0), index_t(spec.n_rows * spec.n_cols));
    update_host(mask_cpu.data(), mask_device.data_handle(), mask_device.extent(0), stream);
    resource::sync_stream(res, stream);
    my_bitmap.set(res, mask_device.view());
    update_host(bitmap_result.data(), my_bitmap.data(), bitmap_result.size(), stream);

    add_cpu_bitmap(bitmap_ref, mask_cpu);
    resource::sync_stream(res, stream);
    ASSERT_TRUE(hostVecMatch(bitmap_ref, bitmap_result, raft::Compare<bitmap_t>()));

    // Flip the bitmap and re-test
    auto bitmap_count = my_bitmap.count(res);
    my_bitmap.flip(res);
    ASSERT_EQ(my_bitmap.count(res), spec.n_rows * spec.n_cols - bitmap_count);
    update_host(bitmap_result.data(), my_bitmap.data(), bitmap_result.size(), stream);
    flip_cpu_bitmap(bitmap_ref);
    resource::sync_stream(res, stream);
    ASSERT_TRUE(hostVecMatch(bitmap_ref, bitmap_result, raft::Compare<bitmap_t>()));

    // Test count() operations
    my_bitmap.reset(res, false);
    ASSERT_EQ(my_bitmap.any(res), false);
    ASSERT_EQ(my_bitmap.none(res), true);
    raft::linalg::range(query_device.data_handle(), query_device.size(), stream);
    my_bitmap.set(res, raft::make_const_mdspan(query_device.view()), true);
    bitmap_count = my_bitmap.count(res);
    ASSERT_EQ(bitmap_count, query_device.size());
    ASSERT_EQ(my_bitmap.any(res), true);
    ASSERT_EQ(my_bitmap.none(res), false);
  }
};

auto inputs_bitmap = ::testing::Values(test_spec_bitmap{4, 8, 5, 10},
                                       test_spec_bitmap{10, 10, 30, 10},
                                       test_spec_bitmap{16, 64, 55, 100},
                                       test_spec_bitmap{100, 100, 1000, 1000},
                                       test_spec_bitmap{16, 2048, 1 << 3, 1 << 12},
                                       test_spec_bitmap{16, 2048, 1 << 24, 1 << 13},
                                       test_spec_bitmap{16, 2048, 1 << 23, 1 << 14});

// using Uint16_32 = BitmapTest<uint16_t, uint32_t>;
// TEST_P(Uint16_32, Run) { run(); }
// INSTANTIATE_TEST_CASE_P(BitmapTest, Uint16_32, inputs_bitmap);

using Uint32_32 = BitmapTest<uint32_t, uint32_t>;
TEST_P(Uint32_32, Run) { run(); }
INSTANTIATE_TEST_CASE_P(BitmapTest, Uint32_32, inputs_bitmap);

// using Uint64_32 = BitmapTest<uint64_t, uint32_t>;
// TEST_P(Uint64_32, Run) { run(); }
// INSTANTIATE_TEST_CASE_P(BitmapTest, Uint64_32, inputs_bitmap);

// using Uint8_64 = BitmapTest<uint8_t, uint64_t>;
// TEST_P(Uint8_64, Run) { run(); }
// INSTANTIATE_TEST_CASE_P(BitmapTest, Uint8_64, inputs_bitmap);

using Uint32_64 = BitmapTest<uint32_t, uint64_t>;
TEST_P(Uint32_64, Run) { run(); }
INSTANTIATE_TEST_CASE_P(BitmapTest, Uint32_64, inputs_bitmap);

// using Uint64_64 = BitmapTest<uint64_t, uint64_t>;
// TEST_P(Uint64_64, Run) { run(); }
// INSTANTIATE_TEST_CASE_P(BitmapTest, Uint64_64, inputs_bitmap);

}  // namespace raft::core
