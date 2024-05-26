#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>


constexpr int tpb = 256;
constexpr int dim = 256;


template <typename value_idx, typename value_t>
__global__ void faster_dot_on_csr_kernel(value_t* __restrict__ dot,
                                         const value_idx* __restrict__ indptr,
                                         const value_idx* __restrict__ cols,
                                         const value_t* __restrict__ A,
                                         const value_t* __restrict__ B,
                                         const value_idx nnz,
                                         const value_idx n_rows,
                                         const value_idx dim)
{
    auto vec_id  = threadIdx.x;
    auto lane_id = threadIdx.x & 0x1f;

    extern __shared__ char smem[];
    value_t* s_A      = (value_t*)smem;
    value_idx cur_row = -1;

    for (int row = blockIdx.x; row < n_rows; row += gridDim.x) {
        for (int dot_id = blockIdx.y + indptr[row]; dot_id < indptr[row + 1]; dot_id += gridDim.y) {
            if (dot_id >= nnz) { return; }
            const value_idx col               = cols[dot_id] * dim;
            const value_t* __restrict__ B_col = B + col;

            if (threadIdx.x == 0) { dot[dot_id] = 0.0; }
            __syncthreads();

            if (cur_row != row) {
                for (value_idx k = vec_id; k < dim; k += blockDim.x) {
                    s_A[k] = A[row * dim + k];
                }
                cur_row = row;
            }

            value_t l_dot_ = 0.0;
            for (value_idx k = vec_id; k < dim; k += blockDim.x) {
                asm("prefetch.global.L2 [%0];" ::"l"(B_col + k + blockDim.x));
                l_dot_ += s_A[k] * __ldcg(B_col + k);
            }
            l_dot_ += __shfl_down_sync(0xffffffff, l_dot_, 16);
            l_dot_ += __shfl_down_sync(0xffff, l_dot_, 8);
            l_dot_ += __shfl_down_sync(0xff, l_dot_, 4);
            l_dot_ += __shfl_down_sync(0xf, l_dot_, 2);
            l_dot_ += __shfl_down_sync(0x3, l_dot_, 1);

            if (lane_id == 0) { atomicAdd_block(dot + dot_id, l_dot_); }
        }
    }
}

template <typename value_idx, typename value_t>
void faster_dot_on_csr(value_t* dot,
                       const value_idx nnz,
                       const value_idx* indptr,
                       const value_idx n_rows,
                       const value_idx n_cols,
                       const value_idx* cols,
                       const value_t* A,
                       const value_t* B,
                       const value_idx dim)
{
    int dev_id, sm_count, blocks_per_sm;
    constexpr value_idx MAX_ROW_PER_ITER = 200;

    const int smem_size = dim * sizeof(value_t);
    cudaGetDevice(&dev_id);
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, faster_dot_on_csr_kernel<value_idx, value_t>, tpb, smem_size);

    auto block_x = std::min(n_rows, MAX_ROW_PER_ITER);
    auto block_y = (std::min(value_idx(blocks_per_sm * sm_count * 16), nnz) + block_x - 1) / block_x;
    dim3 blocks(block_x, block_y, 1);

    std::cout << "tiled blocks_per_sm:" << blocks_per_sm << ", sm_count:" << sm_count
              << ", std::min(value_idx(blocks_per_sm * sm_count), nnz):"
              << std::min(value_idx(blocks_per_sm * sm_count), nnz) << std::endl;
    std::cout << "tiled blocks:" << block_x << ", " << block_y << ", nnz:" << nnz << std::endl;

    faster_dot_on_csr_kernel<value_idx, value_t>
    <<<blocks, tpb, smem_size, 0>>>(dot, indptr, cols, A, B, nnz, n_rows, dim);

    cudaPeekAtLastError();
}

// 主程序
int main()
{
    const int64_t n_rows = 1;
    const int64_t n_cols = 10000000;
    float sparsity       = 0.01;
    int64_t nnz          = (int64_t)(n_rows * n_cols * sparsity);

    int64_t run_times = 1;

    float *A_dev, *B_dev, *C_dev, *dot_dev;
    int64_t *rows_dev, *cols_dev, *indptr_dev;

    float* A_host = new float[n_rows * dim];
    float* B_host = new float[n_cols * dim];

    int64_t* rows_host   = new int64_t[nnz];
    int64_t* cols_host   = new int64_t[nnz];
    int64_t* indptr_host = new int64_t[n_rows + 1];

    // create random maxtri: A, B
    std::srand(std::time(0));
    for (int64_t i = 0; i < n_rows * dim; i++) {
        A_host[i] = 0.01;  // static_cast<float>(rand()) / RAND_MAX;
    }
    for (int64_t i = 0; i < n_cols * dim; i++) {
        B_host[i] = 0.01;  // static_cast<float>(rand()) / RAND_MAX;
    }

    int64_t counter = 0;
    int64_t cur_row = 0;
    indptr_host[0]  = 0;
    for (int64_t i = 0; i < n_rows * n_cols; i++) {
        if (cur_row != i / n_cols) { indptr_host[++cur_row] = counter; }
        int64_t r = static_cast<int64_t>(rand() % 100000);
        if (r < int64_t(sparsity * 100000)) {
            rows_host[counter] = i / n_cols;
            cols_host[counter] = i % n_cols;
            counter++;
        }
        if (counter >= nnz) break;
    }
    indptr_host[n_rows] = counter;

    std::cout << "actual counter: " << counter << ", nnz: " << nnz << std::endl;
    nnz = counter;

    // 分配 GPU 内存
    cudaMalloc(&A_dev, n_rows * dim * sizeof(float));
    cudaMalloc(&B_dev, n_cols * dim * sizeof(float));
    cudaMalloc(&C_dev, n_cols * n_rows * sizeof(float));

    cudaMalloc(&dot_dev, nnz * sizeof(float));
    cudaMalloc(&rows_dev, nnz * sizeof(int64_t));
    cudaMalloc(&cols_dev, nnz * sizeof(int64_t));
    cudaMalloc(&indptr_dev, (n_rows + 1) * sizeof(int64_t));

    // 拷贝数据到 GPU
    cudaMemcpy(A_dev, A_host, n_rows * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B_host, n_cols * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rows_dev, rows_host, nnz * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(cols_dev, cols_host, nnz * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(indptr_dev, indptr_host, (n_rows + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);

    {
        cublasHandle_t handle;
        cublasCreate(&handle);
        std::chrono::time_point<std::chrono::steady_clock> startRecord{};
        std::chrono::time_point<std::chrono::steady_clock> endRecord{};

        float alpha = 1.0f;
        float beta = 0.0f;

        startRecord = std::chrono::steady_clock::now();
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_cols, n_rows, dim, &alpha, B_dev, n_cols, A_dev, dim, &beta, C_dev, n_cols);
        // 等待 GPU 完成
        cudaDeviceSynchronize();
        endRecord = std::chrono::steady_clock::now();


        auto duration_ = std::chrono::duration_cast<std::chrono::nanoseconds>(endRecord - startRecord);
        auto pow_      = static_cast<int32_t>(3) - static_cast<int32_t>(9);
        auto factor    = static_cast<float>(std::pow(10, pow_));
        float latency  = static_cast<float>(duration_.count()) * factor / (float(1.0 * run_times));
        std::cout << "cub latency: " << latency << std::endl;


        cublasDestroy(handle);
    }

    {
        std::chrono::time_point<std::chrono::steady_clock> startRecord{};
        std::chrono::time_point<std::chrono::steady_clock> endRecord{};

        // 调用 kernel
        faster_dot_on_csr<int64_t, float>(
                dot_dev, nnz, indptr_dev, n_rows, n_cols, cols_dev, A_dev, B_dev, dim);
        cudaDeviceSynchronize();

        startRecord = std::chrono::steady_clock::now();
        for (int64_t i = 0; i < run_times; i++) {
            faster_dot_on_csr<int64_t, float>(
                    dot_dev, nnz, indptr_dev, n_rows, n_cols, cols_dev, A_dev, B_dev, dim);
        }
        cudaDeviceSynchronize();
        endRecord = std::chrono::steady_clock::now();

        auto duration_ = std::chrono::duration_cast<std::chrono::nanoseconds>(endRecord - startRecord);
        auto pow_      = static_cast<int32_t>(3) - static_cast<int32_t>(9);
        auto factor    = static_cast<float>(std::pow(10, pow_));
        float latency  = static_cast<float>(duration_.count()) * factor / (float(1.0 * run_times));
        std::cout << "latency: " << latency << std::endl;

        cudaError err = cudaGetLastError();
        if (cudaSuccess != err) { std::cout << cudaGetErrorString(err) << std::endl; }
    }

    // 清理
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
    cudaFree(dot_dev);
    cudaFree(rows_dev);
    cudaFree(rows_dev);
    cudaFree(indptr_dev);
    delete[] A_host;
    delete[] B_host;
    delete[] rows_host;
    delete[] cols_host;
    delete[] indptr_host;

    return 0;
}
