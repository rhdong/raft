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

#include <cublas_v2.h>

#include "cublas_wrappers.hpp"

#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resources.hpp>

namespace raft {
namespace linalg {
namespace detail {

template <typename T, bool DevicePointerMode = false>
void masked_matmul(raft::resources const& handle,
                   T* z,
                   T* x,
                   T* y,
                   int _M,
                   int _N,
                   int _K,
                   bool isZColMajor,
                   bool isXColMajor,
                   bool isYColMajor,
                   cudaStream_t stream,
                   T* alpha,
                   T* beta)
{
  auto cublas_h = raft::resource::get_cublas_handle(handle);
  cublas_device_pointer_mode<DevicePointerMode> pmode(cublas_h);

  cublasOperation_t trans_a, trans_b;
  T *a, *b, *c;
  int lda, ldb, ldc;
  int M, N, K;
  // This function performs c = a * b. Based on the required output layout,
  // either a = x,  b = y or a = y, b = x. In either case c = z.
  if (isZColMajor == true) {
    // Result c is required in column major layout. Thus we perform,
    // z = x * y
    // Using BLAS call c = a * b. Therefore a = x, b = y and c = z

    a = x;
    // If x is in row major layout, cublas needs to transpose x first,
    // therefore trans_x needs to be CUBLAS_OP_T. If x is in column major
    // layout, trans_b needs to be CUBLAS_OP_N.
    trans_a = isXColMajor == true ? CUBLAS_OP_N : CUBLAS_OP_T;
    // Set leading dimension appropriately
    lda = isXColMajor == true ? _M : _K;

    b = y;
    // If y is in row major layout, cublas needs to transpose y first,
    // therefore trans_x needs to be CUBLAS_OP_T. If x is in column major
    // layout, trans_b needs to be CUBLAS_OP_N.
    trans_b = isYColMajor == true ? CUBLAS_OP_N : CUBLAS_OP_T;
    ldb     = isYColMajor == true ? _K : _N;

    c   = z;
    ldc = _M;
    M   = _M;
    N   = _N;
    K   = _K;
  } else {
    // Result c is required in row major layout Thus we pick
    // a = y, b = x and c = a * b = y * x
    // cublas produces output matrix only in column major layout. To get output
    // matrix on row major layout, we need to produce transpose of output
    // in column major layout. Therefore we perform,
    // tr(z) = tr(y) * tr(x)
    // we model this using cublas call for c = a * b
    // therefore a = tr(y), b = tr(x) and c = tr(z)

    a = y;
    // If y is in row major layout, it can be/ interpreted as tr(y) on column
    // major layout. Therefore we can pass trans_a as CUBLAS_OP_N. If y is in
    // column major layout, cublas needs to transpose y first, therefore
    // trans_a needs to be CUBLAS_OP_T
    trans_a = isYColMajor == true ? CUBLAS_OP_T : CUBLAS_OP_N;
    // Set leading dimension appropriately
    lda = isYColMajor == true ? _K : _N;

    b = x;
    // If x is in row major layout, it can be interpreted as tr(x) on column
    // major layout. Therefore we can pass trans_b as CUBLAS_OP_N. If x is in
    // column major layout, cublas needs to trasponse x first, therefore
    // trans_b needs to be CUBLAS_OP_T
    trans_b = isXColMajor == true ? CUBLAS_OP_T : CUBLAS_OP_N;
    // Set leading dimension appropriately
    ldb = isXColMajor == true ? _M : _K;

    c   = z;
    ldc = _N;

    M = _N;
    N = _M;
    K = _K;
  }
  // Actual cuBLAS call
  RAFT_CUBLAS_TRY(
    cublasgemm(cublas_h, trans_a, trans_b, M, N, K, alpha, a, lda, b, ldb, beta, c, ldc, stream));
}

}  // namespace detail
}  // namespace linalg
}  // namespace raft
