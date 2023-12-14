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

#include <cstddef>
#include <cstdint>

#include <raft/core/bitset.cuh>

namespace raft::neighbors::filtering {
/**
 * @brief Filter an index with a bitset
 *
 * @tparam index_t Indexing type
 */
template <typename bitset_t, typename index_t>
struct bitset_filter {
  // View of the bitset to use as a filter
  const raft::core::bitset_view<bitset_t, index_t> bitset_view_;

  bitset_filter(const raft::core::bitset_view<bitset_t, index_t> bitset_for_filtering)
    : bitset_view_{bitset_for_filtering}
  {
  }
  inline _RAFT_HOST_DEVICE bool operator()(
    // query index
    const uint32_t query_ix,
    // the index of the current sample
    const uint32_t sample_ix) const
  {
    return bitset_view_.test(sample_ix);
  }
};

/**
 * @brief The pre-filter for the brute_force algorithm
 *
 * @tparam index_t Indexing type
 */
template <typename index_t>
struct brute_force_filter {
  static constexpr int kNone   = 0;  ///< No filtering mode.
  static constexpr int kBitmap = 1;  ///< Filtering by bitmap mode.

  using bitmap_t = std::uint32_t;

  brute_force_filter() : filter_type{kNone} {}

  brute_force_filter(const raft::core::bitset_view<bitmap_t, index_t> bitmap_for_filtering)
    : brute_force_filter{bitmap_for_filtering}, filter_type{kBitmap}
  {
  }

  inline _RAFT_HOST_DEVICE bool operator()(const index_t sample_id, const index_t query_id) const
  {
    return (filter_type == kNone) || bitmap_view_.test(sample_id, query_id);
  }

 public:
  const int filter_type;

 private:
  const raft::core::bitset_view<bitmap_t, index_t> bitmap_view_;
};

}  // namespace raft::neighbors::filtering
