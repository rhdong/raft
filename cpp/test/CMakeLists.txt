# =============================================================================
# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
# =============================================================================

# ##################################################################################################
# enable testing ################################################################################
# ##################################################################################################
enable_testing()
include(rapids-test)
rapids_test_init()

function(ConfigureTest)

  set(options OPTIONAL LIB EXPLICIT_INSTANTIATE_ONLY NOCUDA)
  set(oneValueArgs NAME GPUS PERCENT)
  set(multiValueArgs PATH TARGETS CONFIGURATIONS)

  cmake_parse_arguments(_RAFT_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(NOT DEFINED _RAFT_TEST_GPUS AND NOT DEFINED _RAFT_TEST_PERCENT)
    set(_RAFT_TEST_GPUS 1)
    set(_RAFT_TEST_PERCENT 30)
  endif()
  if(NOT DEFINED _RAFT_TEST_GPUS)
    set(_RAFT_TEST_GPUS 1)
  endif()
  if(NOT DEFINED _RAFT_TEST_PERCENT)
    set(_RAFT_TEST_PERCENT 100)
  endif()

  if(_RAFT_TEST_NOCUDA)
    set(TEST_NAME "${_RAFT_TEST_NAME}_NOCUDA")
  else()
    set(TEST_NAME ${_RAFT_TEST_NAME})
  endif()

  add_executable(${TEST_NAME} ${_RAFT_TEST_PATH})
  target_link_libraries(
    ${TEST_NAME}
    PRIVATE raft
            raft_internal
            $<$<BOOL:${_RAFT_TEST_LIB}>:raft::compiled>
            GTest::gtest
            GTest::gtest_main
            Threads::Threads
            ${RAFT_CTK_MATH_DEPENDENCIES}
            $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
            $<TARGET_NAME_IF_EXISTS:conda_env>
  )
  set_target_properties(
    ${TEST_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${RAFT_BINARY_DIR}/gtests>"
               INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
  )
  target_compile_options(
    ${TEST_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAFT_CXX_FLAGS}>"
                         "$<$<COMPILE_LANGUAGE:CUDA>:${RAFT_CUDA_FLAGS}>"
  )
  if(_RAFT_TEST_EXPLICIT_INSTANTIATE_ONLY)
    target_compile_definitions(${TEST_NAME} PRIVATE "RAFT_EXPLICIT_INSTANTIATE_ONLY")
  endif()
  if(_RAFT_TEST_NOCUDA)
    target_compile_definitions(${TEST_NAME} PRIVATE "RAFT_DISABLE_CUDA")
  endif()

  target_include_directories(${TEST_NAME} PUBLIC "$<BUILD_INTERFACE:${RAFT_SOURCE_DIR}/test>")

  rapids_test_add(
    NAME ${TEST_NAME}
    COMMAND ${TEST_NAME}
    GPUS ${_RAFT_TEST_GPUS}
    PERCENT ${_RAFT_TEST_PERCENT}
    INSTALL_COMPONENT_SET testing
  )
endfunction()

# ##################################################################################################
# test sources ##################################################################################
# ##################################################################################################

# ##################################################################################################
# * distance tests -------------------------------------------------------------------------

if(BUILD_TESTS)

  ConfigureTest(
    NAME
    SPARSE_TEST
    PATH
    sparse/masked_matmul.cu
  )

endif()

# ##################################################################################################
# Install tests ####################################################################################
# ##################################################################################################
rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/gtests/libraft)
