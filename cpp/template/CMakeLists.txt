# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

# ------------- configure rapids-cmake --------------#

include(cmake/thirdparty/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

# ------------- configure project --------------#

rapids_cuda_init_architectures(test_raft)

project(test_raft LANGUAGES CXX CUDA)

# ------------- configure raft -----------------#

rapids_cpm_init()
include(cmake/thirdparty/get_raft.cmake)

# -------------- compile tasks ----------------- #
add_executable(CAGRA_EXAMPLE src/cagra_example.cu)
target_link_libraries(CAGRA_EXAMPLE PRIVATE raft::raft raft::compiled)

add_executable(IVF_FLAT_EXAMPLE src/ivf_flat_example.cu)
target_link_libraries(IVF_FLAT_EXAMPLE PRIVATE raft::raft raft::compiled)
