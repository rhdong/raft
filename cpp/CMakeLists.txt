# =============================================================================
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
set(RAPIDS_VERSION "24.02")
set(RAFT_VERSION "24.02.00")

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
include(../fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)

option(BUILD_CPU_ONLY "Build CPU only components. Applies to RAFT ANN benchmarks currently" OFF)

# workaround for rapids_cuda_init_architectures not working for arch detection with
# enable_language(CUDA)
set(lang_list "CXX")

if(NOT BUILD_CPU_ONLY)
  include(rapids-cuda)
  rapids_cuda_init_architectures(RAFT)
  list(APPEND lang_list "CUDA")
endif()

project(
  RAFT
  VERSION ${RAFT_VERSION}
  LANGUAGES ${lang_list}
)

# Write the version header
rapids_cmake_write_version_file(include/raft/version_config.hpp)

# ##################################################################################################
# * build type ---------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# this is needed for clang-tidy runs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ##################################################################################################
# * User Options  ------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build raft shared libraries" ON)
option(BUILD_TESTS "Build raft unit-tests" ON)
option(BUILD_PRIMS_BENCH "Build raft C++ benchmark tests" OFF)
option(BUILD_ANN_BENCH "Build raft ann benchmarks" OFF)
option(CUDA_ENABLE_KERNELINFO "Enable kernel resource usage info" OFF)
option(CUDA_ENABLE_LINEINFO
       "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler)" OFF
)
option(CUDA_STATIC_RUNTIME "Statically link the CUDA toolkit runtime and libraries" OFF)
option(CUDA_LOG_COMPILE_TIME "Write a log of compilation times to nvcc_compile_log.csv" OFF)
option(DETECT_CONDA_ENV "Enable detection of conda environment for dependencies" ON)
option(DISABLE_DEPRECATION_WARNINGS "Disable deprecaction warnings " ON)
option(DISABLE_OPENMP "Disable OpenMP" OFF)
option(RAFT_NVTX "Enable nvtx markers" OFF)

set(RAFT_COMPILE_LIBRARY_DEFAULT OFF)
if((BUILD_TESTS
    OR BUILD_PRIMS_BENCH
    OR BUILD_ANN_BENCH
   )
   AND NOT BUILD_CPU_ONLY
)
  set(RAFT_COMPILE_LIBRARY_DEFAULT ON)
endif()
option(RAFT_COMPILE_LIBRARY "Enable building raft shared library instantiations"
       ${RAFT_COMPILE_LIBRARY_DEFAULT}
)

if(BUILD_CPU_ONLY)
  set(BUILD_SHARED_LIBS OFF)
  set(BUILD_TESTS OFF)
endif()

# Needed because GoogleBenchmark changes the state of FindThreads.cmake, causing subsequent runs to
# have different values for the `Threads::Threads` target. Setting this flag ensures
# `Threads::Threads` is the same value across all builds so that cache hits occur
set(THREADS_PREFER_PTHREAD_FLAG ON)

include(CMakeDependentOption)
# cmake_dependent_option( RAFT_USE_FAISS_STATIC "Build and statically link the FAISS library for
# nearest neighbors search on GPU" ON RAFT_COMPILE_LIBRARY OFF )

message(VERBOSE "RAFT: Building optional components: ${raft_FIND_COMPONENTS}")
message(VERBOSE "RAFT: Build RAFT unit-tests: ${BUILD_TESTS}")
message(VERBOSE "RAFT: Building raft C++ benchmarks: ${BUILD_PRIMS_BENCH}")
message(VERBOSE "RAFT: Building ANN benchmarks: ${BUILD_ANN_BENCH}")
message(VERBOSE "RAFT: Build CPU only components: ${BUILD_CPU_ONLY}")
message(VERBOSE "RAFT: Enable detection of conda environment for dependencies: ${DETECT_CONDA_ENV}")
message(VERBOSE "RAFT: Disable depreaction warnings " ${DISABLE_DEPRECATION_WARNINGS})
message(VERBOSE "RAFT: Disable OpenMP: ${DISABLE_OPENMP}")
message(VERBOSE "RAFT: Enable kernel resource usage info: ${CUDA_ENABLE_KERNELINFO}")
message(VERBOSE "RAFT: Enable lineinfo in nvcc: ${CUDA_ENABLE_LINEINFO}")
message(VERBOSE "RAFT: Enable nvtx markers: ${RAFT_NVTX}")
message(VERBOSE
        "RAFT: Statically link the CUDA toolkit runtime and libraries: ${CUDA_STATIC_RUNTIME}"
)

# Set RMM logging level
set(RMM_LOGGING_LEVEL
    "INFO"
    CACHE STRING "Choose the logging level."
)
set_property(
  CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF"
)
message(VERBOSE "RAFT: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

# ##################################################################################################
# * Conda environment detection ----------------------------------------------

if(DETECT_CONDA_ENV)
  rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{CONDA_PREFIX})
    message(
      STATUS "RAFT: No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}"
    )
    set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
  endif()
endif()

# ##################################################################################################
# * compiler options ----------------------------------------------------------

set(_ctk_static_suffix "")
if(CUDA_STATIC_RUNTIME)
  set(_ctk_static_suffix "_static")
endif()

if(NOT BUILD_CPU_ONLY)
  # CUDA runtime
  rapids_cuda_init_runtime(USE_STATIC ${CUDA_STATIC_RUNTIME})
  # * find CUDAToolkit package
  # * determine GPU architectures
  # * enable the CMake CUDA language
  # * set other CUDA compilation flags
  rapids_find_package(
    CUDAToolkit REQUIRED
    BUILD_EXPORT_SET raft-exports
    INSTALL_EXPORT_SET raft-exports
  )
else()
  add_compile_definitions(BUILD_CPU_ONLY)
endif()

if(NOT DISABLE_OPENMP)
  rapids_find_package(
    OpenMP REQUIRED
    BUILD_EXPORT_SET raft-exports
    INSTALL_EXPORT_SET raft-exports
  )
  if(OPENMP_FOUND)
    message(VERBOSE "RAFT: OpenMP found in ${OpenMP_CXX_INCLUDE_DIRS}")
  endif()
endif()

include(cmake/modules/ConfigureCUDA.cmake)

# ##################################################################################################
# * Requirements -------------------------------------------------------------

# add third party dependencies using CPM
rapids_cpm_init()

if(NOT BUILD_CPU_ONLY)
  # thrust before rmm/cuco so we get the right version of thrust/cub
  include(cmake/thirdparty/get_thrust.cmake)
  include(cmake/thirdparty/get_rmm.cmake)
  include(cmake/thirdparty/get_cutlass.cmake)

  include(${rapids-cmake-dir}/cpm/cuco.cmake)
  rapids_cpm_cuco(BUILD_EXPORT_SET raft-exports INSTALL_EXPORT_SET raft-exports)
endif()

if(BUILD_TESTS)
  include(cmake/thirdparty/get_gtest.cmake)
endif()

if(BUILD_PRIMS_BENCH OR BUILD_ANN_BENCH)
  include(${rapids-cmake-dir}/cpm/gbench.cmake)
  rapids_cpm_gbench()
endif()

# ##################################################################################################
# * raft ---------------------------------------------------------------------
add_library(raft INTERFACE)
add_library(raft::raft ALIAS raft)

target_include_directories(
  raft INTERFACE "$<BUILD_INTERFACE:${RAFT_SOURCE_DIR}/include>" "$<INSTALL_INTERFACE:include>"
)

if(NOT BUILD_CPU_ONLY)
  # Keep RAFT as lightweight as possible. Only CUDA libs and rmm should be used in global target.
  target_link_libraries(raft INTERFACE rmm::rmm cuco::cuco nvidia::cutlass::cutlass raft::Thrust)
endif()

target_compile_features(raft INTERFACE cxx_std_17 $<BUILD_INTERFACE:cuda_std_17>)
target_compile_options(
  raft INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-extended-lambda
                 --expt-relaxed-constexpr>
)

set(RAFT_CUSOLVER_DEPENDENCY CUDA::cusolver${_ctk_static_suffix})
set(RAFT_CUBLAS_DEPENDENCY CUDA::cublas${_ctk_static_suffix})
set(RAFT_CURAND_DEPENDENCY CUDA::curand${_ctk_static_suffix})
set(RAFT_CUSPARSE_DEPENDENCY CUDA::cusparse${_ctk_static_suffix})

set(RAFT_CTK_MATH_DEPENDENCIES ${RAFT_CUBLAS_DEPENDENCY} ${RAFT_CUSOLVER_DEPENDENCY}
                               ${RAFT_CUSPARSE_DEPENDENCY} ${RAFT_CURAND_DEPENDENCY}
)

# Endian detection
include(TestBigEndian)
test_big_endian(BIG_ENDIAN)
if(BIG_ENDIAN)
  target_compile_definitions(raft INTERFACE RAFT_SYSTEM_LITTLE_ENDIAN=0)
else()
  target_compile_definitions(raft INTERFACE RAFT_SYSTEM_LITTLE_ENDIAN=1)
endif()

if(RAFT_COMPILE_LIBRARY)
  file(
    WRITE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld"
    [=[
SECTIONS
{
.nvFatBinSegment : { *(.nvFatBinSegment) }
.nv_fatbin : { *(.nv_fatbin) }
}
]=]
  )
endif()

# ##################################################################################################
# * NVTX support in raft -----------------------------------------------------

if(RAFT_NVTX)
  # This enables NVTX within the project with no option to disable it downstream.
  target_link_libraries(raft INTERFACE CUDA::nvToolsExt)
  target_compile_definitions(raft INTERFACE NVTX_ENABLED)
else()
  # Allow enable NVTX downstream if not set here. This creates a new option at build/install time,
  # which is set by default to OFF, but can be enabled in the dependent project.
  get_property(
    nvtx_option_help_string
    CACHE RAFT_NVTX
    PROPERTY HELPSTRING
  )
  string(
    CONCAT
      nvtx_export_string
      "option(RAFT_NVTX \""
      ${nvtx_option_help_string}
      "\" OFF)"
      [=[

target_link_libraries(raft::raft INTERFACE $<$<BOOL:${RAFT_NVTX}>:CUDA::nvToolsExt>)
target_compile_definitions(raft::raft INTERFACE $<$<BOOL:${RAFT_NVTX}>:NVTX_ENABLED>)

  ]=]
  )
endif()

# ##################################################################################################
# * raft_compiled ------------------------------------------------------------
add_library(raft_compiled INTERFACE)

if(TARGET raft_compiled AND (NOT TARGET raft::compiled))
  add_library(raft::compiled ALIAS raft_compiled)
endif()

set_target_properties(raft_compiled PROPERTIES EXPORT_NAME compiled)

if(RAFT_COMPILE_LIBRARY)
  add_library(
    raft_objs OBJECT
    src/core/logger.cpp
    src/distance/detail/pairwise_matrix/dispatch_canberra_double_double_double_int.cu
    src/distance/detail/pairwise_matrix/dispatch_canberra_float_float_float_int.cu
    src/distance/detail/pairwise_matrix/dispatch_correlation_double_double_double_int.cu
    src/distance/detail/pairwise_matrix/dispatch_correlation_float_float_float_int.cu
    src/distance/detail/pairwise_matrix/dispatch_cosine_double_double_double_int.cu
    src/distance/detail/pairwise_matrix/dispatch_cosine_float_float_float_int.cu
    src/distance/detail/pairwise_matrix/dispatch_hamming_unexpanded_double_double_double_int.cu
    src/distance/detail/pairwise_matrix/dispatch_hamming_unexpanded_float_float_float_int.cu
    src/distance/detail/pairwise_matrix/dispatch_hellinger_expanded_double_double_double_int.cu
    src/distance/detail/pairwise_matrix/dispatch_hellinger_expanded_float_float_float_int.cu
    src/distance/detail/pairwise_matrix/dispatch_jensen_shannon_double_double_double_int.cu
    src/distance/detail/pairwise_matrix/dispatch_jensen_shannon_float_float_float_int.cu
    src/distance/detail/pairwise_matrix/dispatch_kl_divergence_double_double_double_int.cu
    src/distance/detail/pairwise_matrix/dispatch_kl_divergence_float_float_float_int.cu
    src/distance/detail/pairwise_matrix/dispatch_l1_double_double_double_int.cu
    src/distance/detail/pairwise_matrix/dispatch_l1_float_float_float_int.cu
    src/distance/detail/pairwise_matrix/dispatch_l2_expanded_double_double_double_int.cu
    src/distance/detail/pairwise_matrix/dispatch_l2_expanded_float_float_float_int.cu
    src/distance/detail/pairwise_matrix/dispatch_l2_unexpanded_double_double_double_int.cu
    src/distance/detail/pairwise_matrix/dispatch_l2_unexpanded_float_float_float_int.cu
    src/distance/detail/pairwise_matrix/dispatch_l_inf_double_double_double_int.cu
    src/distance/detail/pairwise_matrix/dispatch_l_inf_float_float_float_int.cu
    src/distance/detail/pairwise_matrix/dispatch_lp_unexpanded_double_double_double_int.cu
    src/distance/detail/pairwise_matrix/dispatch_lp_unexpanded_float_float_float_int.cu
    src/distance/detail/pairwise_matrix/dispatch_rbf.cu
    src/distance/detail/pairwise_matrix/dispatch_russel_rao_double_double_double_int.cu
    src/distance/detail/pairwise_matrix/dispatch_russel_rao_float_float_float_int.cu
    src/distance/distance.cu
    src/distance/fused_l2_nn.cu
    src/linalg/detail/coalesced_reduction.cu
    src/matrix/detail/select_k_double_int64_t.cu
    src/matrix/detail/select_k_double_uint32_t.cu
    src/matrix/detail/select_k_float_int64_t.cu
    src/matrix/detail/select_k_float_uint32_t.cu
    src/matrix/detail/select_k_float_int32.cu
    src/matrix/detail/select_k_half_int64_t.cu
    src/matrix/detail/select_k_half_uint32_t.cu
    src/neighbors/ball_cover.cu
    src/neighbors/brute_force_fused_l2_knn_float_int64_t.cu
    src/neighbors/brute_force_knn_int64_t_float_int64_t.cu
    src/neighbors/brute_force_knn_int64_t_float_uint32_t.cu
    src/neighbors/brute_force_knn_int_float_int.cu
    src/neighbors/brute_force_knn_uint32_t_float_uint32_t.cu
    src/neighbors/brute_force_knn_index_float.cu
    src/neighbors/detail/cagra/search_multi_cta_float_uint32_dim128_t8.cu
    src/neighbors/detail/cagra/search_multi_cta_float_uint32_dim256_t16.cu
    src/neighbors/detail/cagra/search_multi_cta_float_uint32_dim512_t32.cu
    src/neighbors/detail/cagra/search_multi_cta_float_uint32_dim1024_t32.cu
    src/neighbors/detail/cagra/search_multi_cta_int8_uint32_dim128_t8.cu
    src/neighbors/detail/cagra/search_multi_cta_int8_uint32_dim256_t16.cu
    src/neighbors/detail/cagra/search_multi_cta_int8_uint32_dim512_t32.cu
    src/neighbors/detail/cagra/search_multi_cta_int8_uint32_dim1024_t32.cu
    src/neighbors/detail/cagra/search_multi_cta_uint8_uint32_dim128_t8.cu
    src/neighbors/detail/cagra/search_multi_cta_uint8_uint32_dim256_t16.cu
    src/neighbors/detail/cagra/search_multi_cta_uint8_uint32_dim512_t32.cu
    src/neighbors/detail/cagra/search_multi_cta_uint8_uint32_dim1024_t32.cu
    src/neighbors/detail/cagra/search_single_cta_float_uint32_dim128_t8.cu
    src/neighbors/detail/cagra/search_single_cta_float_uint32_dim256_t16.cu
    src/neighbors/detail/cagra/search_single_cta_float_uint32_dim512_t32.cu
    src/neighbors/detail/cagra/search_single_cta_float_uint32_dim1024_t32.cu
    src/neighbors/detail/cagra/search_single_cta_int8_uint32_dim128_t8.cu
    src/neighbors/detail/cagra/search_single_cta_int8_uint32_dim256_t16.cu
    src/neighbors/detail/cagra/search_single_cta_int8_uint32_dim512_t32.cu
    src/neighbors/detail/cagra/search_single_cta_int8_uint32_dim1024_t32.cu
    src/neighbors/detail/cagra/search_single_cta_uint8_uint32_dim128_t8.cu
    src/neighbors/detail/cagra/search_single_cta_uint8_uint32_dim256_t16.cu
    src/neighbors/detail/cagra/search_single_cta_uint8_uint32_dim512_t32.cu
    src/neighbors/detail/cagra/search_single_cta_uint8_uint32_dim1024_t32.cu
    src/neighbors/detail/ivf_flat_interleaved_scan_float_float_int64_t.cu
    src/neighbors/detail/ivf_flat_interleaved_scan_int8_t_int32_t_int64_t.cu
    src/neighbors/detail/ivf_flat_interleaved_scan_uint8_t_uint32_t_int64_t.cu
    src/neighbors/detail/ivf_flat_search.cu
    src/neighbors/detail/ivf_pq_compute_similarity_float_float.cu
    src/neighbors/detail/ivf_pq_compute_similarity_float_fp8_false.cu
    src/neighbors/detail/ivf_pq_compute_similarity_float_fp8_true.cu
    src/neighbors/detail/ivf_pq_compute_similarity_float_half.cu
    src/neighbors/detail/ivf_pq_compute_similarity_half_fp8_false.cu
    src/neighbors/detail/ivf_pq_compute_similarity_half_fp8_true.cu
    src/neighbors/detail/ivf_pq_compute_similarity_half_half.cu
    src/neighbors/detail/refine_host_float_float.cpp
    src/neighbors/detail/refine_host_int8_t_float.cpp
    src/neighbors/detail/refine_host_uint8_t_float.cpp
    src/neighbors/detail/selection_faiss_int32_t_float.cu
    src/neighbors/detail/selection_faiss_int_double.cu
    src/neighbors/detail/selection_faiss_long_float.cu
    src/neighbors/detail/selection_faiss_size_t_double.cu
    src/neighbors/detail/selection_faiss_size_t_float.cu
    src/neighbors/detail/selection_faiss_uint32_t_float.cu
    src/neighbors/detail/selection_faiss_int64_t_double.cu
    src/neighbors/detail/selection_faiss_int64_t_half.cu
    src/neighbors/detail/selection_faiss_uint32_t_double.cu
    src/neighbors/detail/selection_faiss_uint32_t_half.cu
    src/neighbors/ivf_flat_build_float_int64_t.cu
    src/neighbors/ivf_flat_build_int8_t_int64_t.cu
    src/neighbors/ivf_flat_build_uint8_t_int64_t.cu
    src/neighbors/ivf_flat_extend_float_int64_t.cu
    src/neighbors/ivf_flat_extend_int8_t_int64_t.cu
    src/neighbors/ivf_flat_extend_uint8_t_int64_t.cu
    src/neighbors/ivf_flat_search_float_int64_t.cu
    src/neighbors/ivf_flat_search_int8_t_int64_t.cu
    src/neighbors/ivf_flat_search_uint8_t_int64_t.cu
    src/neighbors/ivfpq_build_float_int64_t.cu
    src/neighbors/ivfpq_build_int8_t_int64_t.cu
    src/neighbors/ivfpq_build_uint8_t_int64_t.cu
    src/neighbors/ivfpq_extend_float_int64_t.cu
    src/neighbors/ivfpq_extend_int8_t_int64_t.cu
    src/neighbors/ivfpq_extend_uint8_t_int64_t.cu
    src/neighbors/ivfpq_search_float_int64_t.cu
    src/neighbors/ivfpq_search_int8_t_int64_t.cu
    src/neighbors/ivfpq_search_uint8_t_int64_t.cu
    src/neighbors/refine_float_float.cu
    src/neighbors/refine_int8_t_float.cu
    src/neighbors/refine_uint8_t_float.cu
    src/raft_runtime/cluster/cluster_cost.cuh
    src/raft_runtime/cluster/cluster_cost_double.cu
    src/raft_runtime/cluster/cluster_cost_float.cu
    src/raft_runtime/cluster/kmeans_fit_double.cu
    src/raft_runtime/cluster/kmeans_fit_float.cu
    src/raft_runtime/cluster/kmeans_init_plus_plus_double.cu
    src/raft_runtime/cluster/kmeans_init_plus_plus_float.cu
    src/raft_runtime/cluster/update_centroids.cuh
    src/raft_runtime/cluster/update_centroids_double.cu
    src/raft_runtime/cluster/update_centroids_float.cu
    src/raft_runtime/distance/fused_l2_min_arg.cu
    src/raft_runtime/distance/pairwise_distance.cu
    src/raft_runtime/matrix/select_k_float_int64_t.cu
    src/raft_runtime/neighbors/brute_force_knn_int64_t_float.cu
    src/raft_runtime/neighbors/cagra_build.cu
    src/raft_runtime/neighbors/cagra_search.cu
    src/raft_runtime/neighbors/cagra_serialize.cu
    src/raft_runtime/neighbors/ivf_flat_build.cu
    src/raft_runtime/neighbors/ivf_flat_search.cu
    src/raft_runtime/neighbors/ivf_flat_serialize.cu
    src/raft_runtime/neighbors/ivfpq_build.cu
    src/raft_runtime/neighbors/ivfpq_deserialize.cu
    src/raft_runtime/neighbors/ivfpq_search_float_int64_t.cu
    src/raft_runtime/neighbors/ivfpq_search_int8_t_int64_t.cu
    src/raft_runtime/neighbors/ivfpq_search_uint8_t_int64_t.cu
    src/raft_runtime/neighbors/ivfpq_serialize.cu
    src/raft_runtime/neighbors/refine_d_int64_t_float.cu
    src/raft_runtime/neighbors/refine_d_int64_t_int8_t.cu
    src/raft_runtime/neighbors/refine_d_int64_t_uint8_t.cu
    src/raft_runtime/neighbors/refine_h_int64_t_float.cu
    src/raft_runtime/neighbors/refine_h_int64_t_int8_t.cu
    src/raft_runtime/neighbors/refine_h_int64_t_uint8_t.cu
    src/raft_runtime/random/rmat_rectangular_generator_int64_double.cu
    src/raft_runtime/random/rmat_rectangular_generator_int64_float.cu
    src/raft_runtime/random/rmat_rectangular_generator_int_double.cu
    src/raft_runtime/random/rmat_rectangular_generator_int_float.cu
    src/spatial/knn/detail/ball_cover/registers_pass_one_2d_dist.cu
    src/spatial/knn/detail/ball_cover/registers_pass_one_2d_euclidean.cu
    src/spatial/knn/detail/ball_cover/registers_pass_one_2d_haversine.cu
    src/spatial/knn/detail/ball_cover/registers_pass_one_3d_dist.cu
    src/spatial/knn/detail/ball_cover/registers_pass_one_3d_euclidean.cu
    src/spatial/knn/detail/ball_cover/registers_pass_one_3d_haversine.cu
    src/spatial/knn/detail/ball_cover/registers_pass_two_2d_dist.cu
    src/spatial/knn/detail/ball_cover/registers_pass_two_2d_euclidean.cu
    src/spatial/knn/detail/ball_cover/registers_pass_two_2d_haversine.cu
    src/spatial/knn/detail/ball_cover/registers_pass_two_3d_dist.cu
    src/spatial/knn/detail/ball_cover/registers_pass_two_3d_euclidean.cu
    src/spatial/knn/detail/ball_cover/registers_pass_two_3d_haversine.cu
    src/spatial/knn/detail/fused_l2_knn_int32_t_float.cu
    src/spatial/knn/detail/fused_l2_knn_int64_t_float.cu
    src/spatial/knn/detail/fused_l2_knn_uint32_t_float.cu
    src/util/memory_pool.cpp
  )
  set_target_properties(
    raft_objs
    PROPERTIES CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
  )

  target_compile_definitions(raft_objs PRIVATE "RAFT_EXPLICIT_INSTANTIATE_ONLY")
  target_compile_options(
    raft_objs PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAFT_CXX_FLAGS}>"
                      "$<$<COMPILE_LANGUAGE:CUDA>:${RAFT_CUDA_FLAGS}>"
  )

  add_library(raft_lib SHARED $<TARGET_OBJECTS:raft_objs>)
  add_library(raft_lib_static STATIC $<TARGET_OBJECTS:raft_objs>)

  set_target_properties(
    raft_lib raft_lib_static
    PROPERTIES OUTPUT_NAME raft
               BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               INTERFACE_POSITION_INDEPENDENT_CODE ON
  )

  foreach(target raft_lib raft_lib_static raft_objs)
    target_link_libraries(
      ${target}
      PUBLIC raft::raft
             ${RAFT_CTK_MATH_DEPENDENCIES} # TODO: Once `raft::resources` is used everywhere, this
                                           # will just be cublas
             $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
    )

    # So consumers know when using libraft.so/libraft.a
    target_compile_definitions(${target} PUBLIC "RAFT_COMPILED")
    # ensure CUDA symbols aren't relocated to the middle of the debug build binaries
    target_link_options(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld")
  endforeach()
endif()

if(TARGET raft_lib AND (NOT TARGET raft::raft_lib))
  add_library(raft::raft_lib ALIAS raft_lib)
endif()

target_link_libraries(raft_compiled INTERFACE raft::raft $<TARGET_NAME_IF_EXISTS:raft::raft_lib>)

# ##################################################################################################
# * raft_compiled_static----------------------------------------------------------------------------

add_library(raft_compiled_static INTERFACE)

if(TARGET raft_compiled_static AND (NOT TARGET raft::compiled_static))
  add_library(raft::compiled_static ALIAS raft_compiled_static)
endif()
set_target_properties(raft_compiled_static PROPERTIES EXPORT_NAME compiled_static)

if(TARGET raft_lib_static AND (NOT TARGET raft::raft_lib_static))
  add_library(raft::raft_lib_static ALIAS raft_lib_static)
endif()

target_link_libraries(
  raft_compiled_static INTERFACE raft::raft $<TARGET_NAME_IF_EXISTS:raft::raft_lib_static>
)

# ##################################################################################################
# * raft_distributed -------------------------------------------------------------------------------
add_library(raft_distributed INTERFACE)

if(TARGET raft_distributed AND (NOT TARGET raft::distributed))
  add_library(raft::distributed ALIAS raft_distributed)
endif()

set_target_properties(raft_distributed PROPERTIES EXPORT_NAME distributed)

rapids_find_generate_module(
  NCCL
  HEADER_NAMES nccl.h
  LIBRARY_NAMES nccl
  BUILD_EXPORT_SET raft-distributed-exports
  INSTALL_EXPORT_SET raft-distributed-exports
)

rapids_export_package(BUILD ucx raft-distributed-exports)
rapids_export_package(INSTALL ucx raft-distributed-exports)
rapids_export_package(BUILD NCCL raft-distributed-exports)
rapids_export_package(INSTALL NCCL raft-distributed-exports)

target_link_libraries(raft_distributed INTERFACE ucx::ucp NCCL::NCCL)

# ##################################################################################################
# * install targets-----------------------------------------------------------
rapids_cmake_install_lib_dir(lib_dir)
include(GNUInstallDirs)
include(CPack)

install(
  TARGETS raft
  DESTINATION ${lib_dir}
  COMPONENT raft
  EXPORT raft-exports
)

install(
  TARGETS raft_compiled raft_compiled_static
  DESTINATION ${lib_dir}
  COMPONENT raft
  EXPORT raft-compiled-exports
)

if(TARGET raft_lib)
  install(
    TARGETS raft_lib
    DESTINATION ${lib_dir}
    COMPONENT compiled
    EXPORT raft-compiled-lib-exports
  )
  install(
    TARGETS raft_lib_static
    DESTINATION ${lib_dir}
    COMPONENT compiled-static
    EXPORT raft-compiled-static-lib-exports
  )
  install(
    DIRECTORY include/raft_runtime
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT compiled
  )
endif()

install(
  TARGETS raft_distributed
  DESTINATION ${lib_dir}
  COMPONENT distributed
  EXPORT raft-distributed-exports
)

install(
  DIRECTORY include/raft
  COMPONENT raft
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Temporary install of raft.hpp while the file is removed
install(
  FILES include/raft.hpp
  COMPONENT raft
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/raft
)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/include/raft/version_config.hpp
  COMPONENT raft
  DESTINATION include/raft
)

# ##################################################################################################
# * install export -----------------------------------------------------------
set(doc_string
    [=[
Provide targets for the RAFT: Reusable Accelerated Functions and Tools

RAFT contains fundamental widely-used algorithms and primitives
for data science and machine learning.

Optional Components:
  - compiled
  - compiled_static
  - distributed

Imported Targets:
  - raft::raft
  - raft::compiled brought in by the `compiled` optional component
  - raft::compiled_static brought in by the `compiled_static` optional component
  - raft::distributed brought in by the `distributed` optional component

]=]
)

set(code_string ${nvtx_export_string})

string(
  APPEND
  code_string
  [=[
if(NOT TARGET raft::Thrust)
  thrust_create_target(raft::Thrust FROM_OPTIONS)
endif()
]=]
)

string(
  APPEND
  code_string
  [=[
if(compiled IN_LIST raft_FIND_COMPONENTS)
  enable_language(CUDA)
endif()
]=]
)
set(raft_components compiled distributed)
set(raft_export_sets raft-compiled-exports raft-distributed-exports)
if(TARGET raft_lib)
  list(APPEND raft_components compiled compiled-static)
  list(APPEND raft_export_sets raft-compiled-lib-exports raft-compiled-static-lib-exports)
endif()

string(
  APPEND
  code_string
  [=[
 option(RAFT_ENABLE_CUSOLVER_DEPENDENCY "Enable cusolver dependency" ON)
 option(RAFT_ENABLE_CUBLAS_DEPENDENCY "Enable cublas dependency" ON)
 option(RAFT_ENABLE_CURAND_DEPENDENCY "Enable curand dependency" ON)
 option(RAFT_ENABLE_CUSPARSE_DEPENDENCY "Enable cusparse dependency" ON)

mark_as_advanced(RAFT_ENABLE_CUSOLVER_DEPENDENCY)
mark_as_advanced(RAFT_ENABLE_CUBLAS_DEPENDENCY)
mark_as_advanced(RAFT_ENABLE_CURAND_DEPENDENCY)
mark_as_advanced(RAFT_ENABLE_CUSPARSE_DEPENDENCY)

target_link_libraries(raft::raft INTERFACE
  $<$<BOOL:${RAFT_ENABLE_CUSOLVER_DEPENDENCY}>:${RAFT_CUSOLVER_DEPENDENCY}>
  $<$<BOOL:${RAFT_ENABLE_CUBLAS_DEPENDENCY}>:${RAFT_CUBLAS_DEPENDENCY}>
  $<$<BOOL:${RAFT_ENABLE_CUSPARSE_DEPENDENCY}>:${RAFT_CUSPARSE_DEPENDENCY}>
  $<$<BOOL:${RAFT_ENABLE_CURAND_DEPENDENCY}>:${RAFT_CURAND_DEPENDENCY}>
)
]=]
)

# Use `rapids_export` for 22.04 as it will have COMPONENT support
rapids_export(
  INSTALL raft
  EXPORT_SET raft-exports
  COMPONENTS ${raft_components}
  COMPONENTS_EXPORT_SET ${raft_export_sets}
  GLOBAL_TARGETS raft compiled distributed
  NAMESPACE raft::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string
)

# ##################################################################################################
# * build export -------------------------------------------------------------
rapids_export(
  BUILD raft
  EXPORT_SET raft-exports
  COMPONENTS ${raft_components}
  COMPONENTS_EXPORT_SET ${raft_export_sets}
  GLOBAL_TARGETS raft compiled distributed
  DOCUMENTATION doc_string
  NAMESPACE raft::
  FINAL_CODE_BLOCK code_string
)

# ##################################################################################################
# * shared test/bench headers ------------------------------------------------

if(BUILD_TESTS OR BUILD_PRIMS_BENCH)
  include(internal/CMakeLists.txt)
endif()

# ##################################################################################################
# * build test executable ----------------------------------------------------

if(BUILD_TESTS)
  include(test/CMakeLists.txt)
endif()

# ##################################################################################################
# * build benchmark executable -----------------------------------------------

if(BUILD_PRIMS_BENCH)
  include(bench/prims/CMakeLists.txt)
endif()

# ##################################################################################################
# * build ann benchmark executable -----------------------------------------------

if(BUILD_ANN_BENCH)
  include(bench/ann/CMakeLists.txt)
endif()
